"""Module to memoize function results on disk with python code and data dependencies tracking."""

import base64
import binascii
import builtins
import contextlib
import copy
import datetime
import functools
import gc
import hashlib
import inspect
import json
import os
import pickle
import stat
import sys
import tempfile
import time
import types
from dataclasses import asdict, dataclass
from dis import HAVE_ARGUMENT, opmap
from sysconfig import get_path
from importlib import import_module
from os.path import exists
from types import ModuleType
from typing import (
    IO,
    Any,
    Callable,
    Dict,
    Generator,
    Generic,
    List,
    Optional,
    Set,
    Tuple,
    TypeVar,
)

from typing_extensions import ParamSpec


def get_python_lib() -> str:
    """Get the path to the Python library."""
    return get_path("purelib")  # type: ignore


# numpy as numba used to get the random states before and after function call
# could use plugin approach instead
np: Optional[ModuleType]
try:
    import numpy as np
except ImportError:
    np = None


max_bytes = 2**31 - 1
# disk_cache_dir = os.path.join(tempfile.gettempdir(), "memodisk_cache")
disk_cache_dir = os.path.join(os.path.abspath(os.path.dirname(sys.argv[0])), "__memodisk__")
user_ignore_files: Set[str] = set()  # should contain / not \
ignore_folders = [os.path.dirname(os.__file__), get_python_lib()]
strings_filter_out = {
    os.path.normpath(os.path.join("python", "debugpy", "_vendored", "pydevd")),
    "__pycache__",
}
skip_check_global_changed = {"<class 'numba.core.registry.CPUDispatcher'>"}

# helps to check caching loading worked in tests
__last_cache_loading__: Optional[str] = None


@dataclass(frozen=True)
class FuncKeyType:
    """Dataclass describing a function uniquely used as key in some dictionaries"""

    func_name: str
    filename: str
    line_number: int


def get_numba_dispatcher_hash(x: Any) -> str:
    return hashlib.sha256(x.__code__.co_code).hexdigest()


hashing_func_map: Dict[Any, Callable[[Any], str]] = {
    str: str,
    int: str,
    types.FunctionType: lambda x: x.__qualname__,
    types.MethodType: lambda x: x.__qualname__,
    "<class 'numba.core.registry.CPUDispatcher'>": get_numba_dispatcher_hash,
}


def get_last_cache_loading() -> Optional[str]:
    return __last_cache_loading__


def reset_last_cache_loading() -> None:
    global __last_cache_loading__
    __last_cache_loading__ = None


@dataclass(frozen=True)  # frozen so that it is hashable and can be used a dict key
class DataDependency:
    """Dataclass to store information on data dependency"""

    file_path: str
    last_modified_date_str: str


@dataclass(frozen=True)
class CodeDependency:
    """Dataclass to store information on code dependency."""

    function_qualified_name: str
    module: Optional[str]
    filename: str
    bytecode_hash: str
    global_vars: Dict[str, str]
    closure_vars: Dict[str, str]


@dataclass(frozen=True)
class FunctionDependencies:
    """Dataclass to store information on code and data dependency"""

    code: list
    data: list
    inherited: list
    random_states: Optional[dict]


functions_hash_cache: Dict[FuncKeyType, Tuple[List[str], str]] = {}
functions_path_to_hash_cache: Dict[str, str] = {}

builtins_open = builtins.open
open_delay = 0.001


GlobalsType = Dict[FuncKeyType, Dict[str, str]]


def set_cache_dir(folder: str) -> None:
    global disk_cache_dir
    disk_cache_dir = folder


def pickle_big_data(data: Any, file_path: str) -> None:
    bytes_out = pickle.dumps(data, protocol=4)
    with builtins_open(file_path, "wb") as f_out:
        for idx in range(0, len(bytes_out), max_bytes):
            f_out.write(bytes_out[idx : idx + max_bytes])


def unpickle_big_data(file_path: str) -> Any:
    bytes_in = bytearray(0)
    input_size = os.path.getsize(file_path)
    with builtins_open(file_path, "rb") as f_in:
        for _ in range(0, input_size, max_bytes):
            bytes_in += f_in.read(max_bytes)

    return pickle.loads(bytes_in)


def get_globals_from_code(code: types.CodeType) -> List[str]:
    global_ops = opmap["LOAD_GLOBAL"], opmap["STORE_GLOBAL"]
    extended_args = opmap["EXTENDED_ARG"]

    names = code.co_names
    op = (int(c) for c in code.co_code)
    global_var_names = set()

    extarg = 0

    for c in op:
        if c in global_ops:
            global_var_names.add(names[next(op) + extarg])

        elif c == extended_args:
            continue
        elif c == opmap["CALL_FUNCTION"]:
            pass
        elif c >= HAVE_ARGUMENT:
            next(op)

        extarg = 0

    return sorted(global_var_names)


def get_function_from_frame(frame: types.FrameType) -> Optional[Callable]:
    # inspired from https://stackoverflow.com/questions/4492559/how-to-get-current-function-into-a-variable
    code = frame.f_code
    globs = frame.f_globals
    functype = type(lambda: 0)

    if "self" in frame.f_locals:  # rely on convention it is called self
        self = frame.f_locals["self"]
        for name in dir(self):
            func = getattr(self, name, None)
            if isinstance(func, types.MethodType):
                if getattr(func, "__code__", None) is code:
                    if getattr(func, "__globals__", None) is globs:
                        return func

    for func in gc.get_referrers(code):
        if type(func) is functype:
            if getattr(func, "__code__", None) is code:
                return func
                # numba tests fail if we use following lines instead, not sure why
                # if getattr(func, "__globals__", None) is globs: # numpy gfa
                #     return func
    if frame.f_back is not None:
        for func in frame.f_back.f_globals:
            if type(func) is functype:
                if getattr(func, "__code__", None) is code:
                    return func
    return None


def get_function_qualified_name_from_frame(frame: types.FrameType) -> str:
    func = get_function_from_frame(frame)
    assert func is not None, (
        f"Did not find function {frame.f_code}.\n"
        "It could be due to presence of a debugging breakpoint "
        "in this function that causes the bytecode to differ from the "
        "bytecode of the function referred to in the calling function."
    )
    return func.__qualname__


def get_hash(variable: Any) -> str:
    return hashlib.sha256(pickle.dumps(variable)).hexdigest()


def get_global_hash(
    name: str,
    variable: Any,
    frame: Optional[types.FrameType],
    co: Optional[types.CodeType],
) -> str:
    variable_type = type(variable)
    if str(variable_type) in hashing_func_map:
        hash_str = hashing_func_map[str(variable_type)](variable)
    elif variable_type in hashing_func_map:
        hash_str = hashing_func_map[variable_type](variable)
    else:
        try:
            pickled_var = pickle.dumps(variable)
        except BaseException as e:
            if frame is None:
                raise BaseException(f"Could not pickle global variable {name}") from e
            else:
                assert co is not None
                raise BaseException(
                    f"Could not pickle global variable {name} used in function {co.co_name} in "
                    f"{co.co_filename}, line {frame.f_code.co_firstlineno}. {ignore_folders}: {e}"
                ) from e
        hash_str = hashlib.sha256(pickled_var).hexdigest()
    return hash_str


def get_bytecode_hash(code: types.CodeType) -> str:
    # remove code object constants as this get covered
    consts = []
    for const in code.co_consts:
        if not isinstance(const, type(code)):
            consts.append(const)
        else:
            # not sure this is enough
            consts.append((const.co_name, const.co_filename))
    return hashlib.sha256(pickle.dumps(consts) + code.co_code).hexdigest()


class Tracer:
    """Class used to track all dependencies of the functions.
    This is intended to be used as a singleton. A single class instance will track dependencies for all
    memoized functions.
    This is to avoid cascading callbacks when calling the trace function when we have nested memoized functions.
    This instance counts how many time a function is called and a dta file accessed.
    To get the dependencies of a function we then need to compare the counters before and after the function execution
    """

    def __init__(self) -> None:
        self.is_registered = False
        self.tracing_activated = False
        self.clear_counters()
        self._ignore_files = [
            os.path.normpath(__file__),
            "<frozen importlib._bootstrap_external>",
            "<frozen importlib._bootstrap>",
            "<__array_function__ internals>",
            "<frozen zipimport>",
            "<string>",
            "<attrs generated init _pytest._code.code.FormattedExcinfo>",
            "<attrs generated init _pytest._code.code.ReprFuncArgs>",
            "<attrs generated init _pytest._code.code.ReprFileLocation>",
            "<attrs generated init _pytest._code.code.ReprEntry>",
            "<attrs generated init _pytest._code.code.ReprTraceback>",
            "<attrs generated init _pytest._code.code.ExceptionChainRepr>",
        ]

    def clear_counters(self) -> None:
        self.code_dependencies_counters: Dict[FuncKeyType, int] = {}
        self.function_alias: Dict[FuncKeyType, str] = {}
        self.function_qualified_name: Dict[FuncKeyType, str] = {}
        self.function_bytecode_hash: Dict[FuncKeyType, str] = {}
        self.function_modules: Dict[FuncKeyType, Optional[str]] = {}
        self.data_dependencies_counters: Dict[DataDependency, int] = {}
        self.inherited_dependencies_counters: Dict[DataDependency, int] = {}
        self.globals: GlobalsType = {}
        self.closure_vars: GlobalsType = {}

    def register(self) -> None:
        assert not self.is_registered
        self.prevtrace = sys.getprofile()
        sys.setprofile(self.tracefunc)
        self.is_registered = True
        self.tracing_activated = True

    def unregister(self) -> None:
        assert self.is_registered
        self.tracing_activated = False
        sys.setprofile(self.prevtrace)
        self.is_registered = False

    def add_data_dependency(self, file_path: str) -> None:
        file_path = os.path.abspath(file_path)
        if any(file_path.lower().startswith(folder.lower()) for folder in ignore_folders):
            return
        for string_filter_out in strings_filter_out:
            if string_filter_out in file_path:
                return
        # we are assuming we cannot modify a file within the duration precision of st_mtime
        last_modified_date = os.stat(file_path).st_mtime
        last_modified_date_str = str(datetime.datetime.fromtimestamp(last_modified_date))
        dep = DataDependency(file_path=file_path, last_modified_date_str=last_modified_date_str)
        if dep in self.data_dependencies_counters:
            self.data_dependencies_counters[dep] += 1
        else:
            self.data_dependencies_counters[dep] = 1

    def tracefunc(self, frame: types.FrameType, event: str, arg: str) -> None:
        # putting breakpoints in this method does not work
        if not self.tracing_activated:
            # if self.prevtrace is not None:
            #     self.prevtrace(frame, event, arg)
            # probably with the sys.settrace call in unregister
            # need to return here to avoid adding debugpy
            # functions into the list of dependencies
            # when debugging in visual studio code
            return
        if event == "call":
            co = frame.f_code
            filename = os.path.normpath(co.co_filename)
            function_name = co.co_name
            line_number = frame.f_code.co_firstlineno
            if filename == __file__:
                return
            func_key = FuncKeyType(
                func_name=function_name,
                filename=filename,
                line_number=line_number,
            )
            if (
                not any(filename.lower().startswith(folder.lower()) for folder in ignore_folders)
                and filename not in self._ignore_files
                and function_name
                not in [
                    "<genexpr>",
                    "<listcomp>",
                    "<module>",
                ]
                and filename not in user_ignore_files
            ):
                for string_filter_out in strings_filter_out:
                    if string_filter_out in filename:
                        return

                if func_key not in self.function_bytecode_hash:
                    self.function_bytecode_hash[func_key] = get_bytecode_hash(co)

                # check subfunction is pure
                if func_key not in self.globals:
                    globals_candidates = get_globals_from_code(co)
                    closure_var_names_candidates = co.co_freevars
                    self.globals[func_key] = {}
                    self.closure_vars[func_key] = {}
                    for name in globals_candidates:
                        if (
                            (name not in frame.f_builtins)
                            and (name in frame.f_globals)
                            and (
                                type(frame.f_globals[name])
                                not in [
                                    type,
                                    types.ModuleType,
                                    types.BuiltinFunctionType,
                                ]
                            )
                        ):
                            variable = frame.f_globals[name]
                            hash_str = get_global_hash(name, variable, frame, co)
                            # TODO should use hash here ?
                            self.globals[func_key][name] = hash_str

                    for name in closure_var_names_candidates:
                        if (name not in frame.f_builtins) and (
                            type(frame.f_locals[name])
                            not in [
                                type,
                                types.ModuleType,
                                types.BuiltinFunctionType,
                            ]
                        ):
                            variable = frame.f_locals[name]
                            hash_str = get_global_hash(name, variable, frame, co)
                            # TODO should use hash here ?
                            self.closure_vars[func_key][name] = hash_str
                else:
                    for name, hash_str in self.globals[func_key].items():
                        variable = frame.f_globals[name]
                        if str(type(variable)) not in skip_check_global_changed:
                            new_hash_str = get_global_hash(name, variable, frame, co)

                            if new_hash_str != hash_str:
                                raise BaseException(
                                    f'Global variable {name} used in function {co.co_name} in "{filename}",'
                                    f" line {frame.f_code.co_firstlineno} changed during two calls."
                                )
                if func_key not in self.function_qualified_name:
                    self.function_qualified_name[func_key] = get_function_qualified_name_from_frame(frame)

                if func_key not in self.function_alias:
                    if func_key not in self.function_modules:
                        module = inspect.getmodule(frame)
                        assert module is not None, f"Module not found for function {function_name} in {filename}"
                        self.function_modules[func_key] = module.__name__

                if func_key in self.code_dependencies_counters:
                    self.code_dependencies_counters[func_key] += 1
                else:
                    self.code_dependencies_counters[func_key] = 1

        # Calling the previous trace handler to keep the debugger happy
        if self.prevtrace is not None:
            self.tracing_activated = False
            self.prevtrace(frame, event, arg)
            self.tracing_activated = True


tracer = Tracer()


class FunctionNotFoundError(Exception):
    """Custom error when function not found."""

    pass


def class_from_class_str(
    string: str,
) -> Any:
    path_name = string.rsplit(".", 1)
    if len(path_name) == 2:
        path, name = path_name
        try:
            return getattr(import_module(path), name)
        except ModuleNotFoundError:
            path_name2 = path.rsplit(".", 1)
            if len(path_name2) == 2:
                path2, name2 = path_name2
                return getattr(getattr(import_module(path2), name2), name)
            raise
    else:
        return __builtins__[string]  # type: ignore


def print_change(message: str) -> None:
    print("*****************")
    print("Memoization miss:")
    print(message)
    print("*****************")


def dependency_changed(func: Callable, all_dependencies: dict) -> bool:
    """Detect if any of the dependencies of the function has changed."""
    # detect if any random state changed
    if all_dependencies["random_states"] is not None:
        assert np is not None, "You need the numpy module"
        random_state_before = {"numpy": base64.b64encode(pickle.dumps(np.random.get_state())).decode("utf-8")}
        for name in ["numpy"]:
            if not random_state_before[name] == all_dependencies["random_states"]["before"][name]:
                print_change("random seed changed")
                return True

    # detect if any data dependency changed
    for entry_dict in all_dependencies["data"]:
        entry_data = DataDependency(**entry_dict)
        filepath = entry_data.file_path
        if not os.path.exists(filepath):
            print_change(f"Data dependency {filepath} not found")
            return True
        last_modified_date_str = str(datetime.datetime.fromtimestamp(os.stat(filepath).st_mtime))
        if entry_data.last_modified_date_str != last_modified_date_str:
            print_change(
                f"Data dependency {filepath} has been modified."
                f"Memoize with {entry_data.last_modified_date_str}, now {last_modified_date_str}"
            )
            return True

    # detect if any code dependency or used global variable changed
    for entry_dict in all_dependencies["code"]:
        entry_code = CodeDependency(**entry_dict)
        function_qualified_name = entry_code.function_qualified_name
        filename = entry_code.filename

        filtered_out = False
        for string_filter_out in strings_filter_out:
            if string_filter_out in filename:
                print(f"Dependencies of file {filename} filtered out")
                filtered_out = True
                continue
        if filtered_out:
            continue
        bytecode_hash = entry_code.bytecode_hash
        global_variables = entry_code.global_vars
        closure_variables = entry_code.closure_vars
        if not os.path.exists(filename):
            print_change(f"Could not find file {filename}.")
            return True

        # retrieve handle to the function
        func_dep = func
        if func.__qualname__ != function_qualified_name:
            if entry_code.module == "__main__":
                node = inspect.getmodule(func)
            elif entry_code.module is not None:
                # get function from module name and function qualified name
                node = import_module(entry_code.module)
            else:
                node = inspect.getmodule(func)

            for name in function_qualified_name.split("."):
                new_node = getattr(node, name, None)
                if new_node is None:
                    assert node is not None
                    if getattr(node, "__qualname__", None):
                        print_change(f"Could not find {name} in {node.__qualname__} in {filename}")
                    else:
                        print_change(f"Could not find {name} in {node} in {filename}")

                    return True
                node = new_node
            if isinstance(node, property):
                # TODO might need to store in the json if getter or setter:
                node = node.fget
            assert callable(node)
            func_dep = node

            try:
                if func_dep.__code__.co_name == "_memoize_wrapper":
                    func_dep = func_dep.__wrapped__
            except Exception:
                return True

            if not func_dep.__qualname__ == function_qualified_name:
                print_change(f"Function {function_qualified_name} not found")
                return True
        # check global variables did not change
        dep_global_vars = getattr(func_dep, "__globals__", None)
        # assert dep_global_vars is not None # does not work for numba CPU Dispatcher
        for name, value in global_variables.items():
            if dep_global_vars is None or name not in dep_global_vars:
                print_change(f"Global variable {name} used in {function_qualified_name} not found")
                return True
            variable = dep_global_vars[name]
            hash_str = get_global_hash(name, variable, None, None)
            if hash_str != value:
                print_change(f"Global variable {name} used in {function_qualified_name} has been modified")
                return True

        # check closure variables did not change

        for name, value in closure_variables.items():
            dep_closure_vars = inspect.getclosurevars(func_dep).nonlocals
            variable = dep_closure_vars[name]
            hash_str = get_global_hash(name, variable, None, None)
            if hash_str != value:
                print_change(f"Closure variable {name} used in {function_qualified_name} has been modified")
                return True
        try:
            new_bytecode_hash = get_bytecode_hash(func_dep.__code__)
            if new_bytecode_hash != bytecode_hash:
                print_change(
                    f"Code dependency {function_qualified_name} has been modified."
                    f" Was {bytecode_hash} now {new_bytecode_hash}"
                )
                return True
        except FunctionNotFoundError as e:
            print_change(str(e))
    return False


def get_dependencies_runtime(func: Callable, *args: Any, **kwargs: Any) -> Tuple[Any, FunctionDependencies]:
    if not tracer.is_registered:
        tracer.clear_counters()
        tracer.register()
        unregister_after_execution = True
        # replace build in open function with wrapper that keeps track of files that a accessed for reading
        prev_open = builtins.open

        builtins.open = memoized_open_wrapper  # type: ignore
    else:
        # we use the memoization decorator on nested functions
        unregister_after_execution = False
    code_dependencies_counters_copy = copy.copy(tracer.code_dependencies_counters)
    data_dependencies_counters_copy = copy.copy(tracer.data_dependencies_counters)
    inherited_dependencies_counters_copy = copy.copy(tracer.inherited_dependencies_counters)
    frame = inspect.currentframe()
    assert frame is not None

    if np is not None:
        random_state_before = {
            "numpy": base64.b64encode(pickle.dumps(np.random.get_state())).decode("utf-8"),
        }
    else:
        random_state_before = None

    result = func(*args, **kwargs)
    if np is not None:
        random_state_after = {
            "numpy": base64.b64encode(pickle.dumps(np.random.get_state())).decode("utf-8"),
        }
    else:
        random_state_after = None
    random_state_changed = False
    if np is not None:
        assert random_state_after is not None
        assert random_state_before is not None
        for name in ["numpy"]:
            if not random_state_before[name] == random_state_after[name]:
                random_state_changed = True
        if random_state_changed:
            random_states = {"before": random_state_before, "after": random_state_after}
        else:
            random_states = None
    if unregister_after_execution:
        # restore open function
        tracer.unregister()
        builtins.open = prev_open

    tracer.tracing_activated = False
    code_dependencies_list = []
    all_qual_names = set()
    for func_key, val in tracer.code_dependencies_counters.items():
        if func_key in code_dependencies_counters_copy and code_dependencies_counters_copy[func_key] == val:
            continue
        filename = func_key.filename
        # last_modified_date = os.path.getmtime(filename)
        function_qualified_name = tracer.function_qualified_name[func_key].split(".")
        bytecode_hash = tracer.function_bytecode_hash[func_key]

        # remove nested function if parent already listed as dependency
        if len(function_qualified_name) > 1:
            covered = False
            for k in range(len(function_qualified_name)):
                parent_name = filename + "." + ".".join(function_qualified_name[:-k])
                if parent_name in all_qual_names:
                    covered = True
                    break
            if covered:
                continue

        module = tracer.function_modules[func_key]
        assert module is not None
        function_qualified_name_str = ".".join(function_qualified_name)
        all_qual_names.add(filename + "." + function_qualified_name_str)

        code_dependencies_list.append(
            CodeDependency(
                function_qualified_name=function_qualified_name_str,
                module=module,
                filename=filename,
                bytecode_hash=bytecode_hash,
                global_vars=tracer.globals[func_key],
                closure_vars=tracer.closure_vars[func_key],
            )
        )

    data_dependencies_list = []
    inherited_dependencies_list = []

    for key, val in tracer.data_dependencies_counters.items():
        if key in data_dependencies_counters_copy and data_dependencies_counters_copy[key] == val:
            continue
        data_dependencies_list.append(key)

    for key, val in tracer.inherited_dependencies_counters.items():
        if key in inherited_dependencies_counters_copy and inherited_dependencies_counters_copy[key] == val:
            continue
        inherited_dependencies_list.append(key)

    tracer.tracing_activated = True
    dependencies = FunctionDependencies(
        code=code_dependencies_list,
        data=data_dependencies_list,
        inherited=inherited_dependencies_list,
        random_states=random_states,
    )
    return result, dependencies


P = ParamSpec("P")
R = TypeVar("R")


def memoize(func: Callable[P, R]) -> Callable[P, R]:
    @functools.wraps(func)
    def _memoize_wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        global disk_cache_dir
        co = func.__code__

        full_qualified_name = f"{func.__module__}.{func.__qualname__}"
        full_qualified_name_file = full_qualified_name.replace("<locals>", "locals")

        hash_str = hashlib.sha256(
            pickle.dumps(
                (
                    co.co_name,
                    co.co_filename,
                    args,
                    kwargs,
                    inspect.getclosurevars(func).nonlocals,
                )
            )
        ).hexdigest()
        hash_str_short = hash_str[:16]
        result_file = os.path.join(disk_cache_dir, f"{full_qualified_name_file}_{hash_str_short}_result.pkl")
        dependencies_file = os.path.join(
            disk_cache_dir,
            f"{full_qualified_name_file}_{hash_str_short}_dependencies.json",
        )

        assert len(dependencies_file) < 260, "long file names no handled yet"
        use_cached = False
        result: R

        if exists(dependencies_file):
            use_cached = True
            with builtins_open(dependencies_file, "r") as fh:
                all_dependencies = json.load(fh)
            use_cached = not dependency_changed(func, all_dependencies)

            if use_cached:
                print(f"Result loaded from {os.path.split(result_file)[1]}")
                try:
                    result = unpickle_big_data(result_file)
                except Exception:
                    use_cached = False
                    print(
                        f"WARNING: Could not unpickle the data from {os.path.split(result_file)[1]},"
                        " probably due to code change."
                    )

            if use_cached:
                global __last_cache_loading__
                __last_cache_loading__ = full_qualified_name_file
                # increase the counter for the functions used by the
                # function whose results is loaded from the cache
                last_modified_date = os.stat(dependencies_file).st_mtime
                last_modified_date_str = str(datetime.datetime.fromtimestamp(last_modified_date))
                dep = DataDependency(
                    file_path=dependencies_file,
                    last_modified_date_str=last_modified_date_str,
                )
                if dep in tracer.inherited_dependencies_counters:
                    tracer.inherited_dependencies_counters[dep] += 1
                else:
                    tracer.inherited_dependencies_counters[dep] = 1

                if np is not None:
                    if all_dependencies["random_states"] is not None:
                        np.random.set_state(
                            pickle.loads(
                                binascii.a2b_base64(all_dependencies["random_states"]["after"]["numpy"].encode("utf-8"))
                            )
                        )
                return result
        result, dependencies = get_dependencies_runtime(func, *args, **kwargs)

        # could skip if the expected time to save is larger than the time to compute
        os.makedirs(os.path.dirname(result_file), exist_ok=True)
        pickle_big_data(result, result_file)

        assert len(dependencies.code) > 0
        all_dependencies = {
            "arguments_hash": hash_str,
            "code": [asdict(d) for d in dependencies.code],
            "data": [asdict(d) for d in dependencies.data],
            "random_states": dependencies.random_states,
        }
        for inherited_dependencies in dependencies.inherited:
            with builtins_open(inherited_dependencies.file_path, "r") as fh:
                dependencies_to_add = json.load(fh)
            for data_dep in dependencies_to_add["data"]:
                # TODO add only of not already in the list,
                # could use set of frozen dataclass
                all_dependencies["data"].append(data_dep)
            for data_dep in dependencies_to_add["code"]:
                # TODO add only of not already in the list,
                # could use set of frozen dataclass
                all_dependencies["code"].append(data_dep)

        with builtins_open(dependencies_file, "w") as fh:
            json.dump(all_dependencies, fh, indent=4)

        return result  # ty

    return _memoize_wrapper


def add_data_dependency(filename: str) -> None:
    """Empty function used to specify dependency on some data file or executable.

    Calls to this function are detected and the last modification
    date of the file is used to detect changes.
    """
    tracer.add_data_dependency(filename)


@contextlib.contextmanager
def loop_until_access_time_greater_than_modification_time(
    filename: str, verbose: bool = False
) -> Generator[None, None, None]:
    """Make the file read only to get the modification date and wait long enough
    to prevent the file to be modified by another process within
    the time interval where the modification
    date remains unchanged due to time precision.
    """
    chmode = os.stat(filename).st_mode
    ro_mask = 0o777 ^ (stat.S_IWRITE | stat.S_IWGRP | stat.S_IWOTH)
    os.chmod(filename, chmode & ro_mask)
    if verbose:
        print(f"Locking {filename}")
    tracer.add_data_dependency(filename)
    file_last_modification = os.stat(filename).st_mtime

    yield None

    while True:
        # loop to make sure we wait long enough so that the same file does not get
        # modified twice during the time interval during which the modification time
        # remain the same due to the limited modification time precision
        # this might not work if the modification is done by another process
        # We create a temporary file and check its last modification date is strictly greater
        # than the file we are reading in this function
        with tempfile.NamedTemporaryFile() as fp:
            fp.write(b"dummy")
            tmp_file_last_modification = os.stat(fp.name).st_mtime

        if tmp_file_last_modification == file_last_modification:
            if verbose:
                print("Waiting.")
            time.sleep(open_delay)
        else:
            break
    if verbose:
        print(f"Release {filename}")
    os.chmod(filename, chmode)


def memoized_open_wrapper(filename: str, mode: str = "r", encoding: Optional[str] = None) -> IO[Any]:
    if mode in ["r", "rb"]:
        if not os.path.exists(filename):
            raise FileNotFoundError

        with loop_until_access_time_greater_than_modification_time(filename):
            out = builtins_open(filename, mode, encoding=encoding)
            return out
    else:
        out = builtins_open(filename, mode, encoding=encoding)
        return out


RetType = TypeVar("RetType")


class DataLoaderWrapper(Generic[RetType]):
    """Wrap the function for the input file to be added as data dependency."""

    def __init__(self, fun: Callable[..., RetType], position: int = 0):
        self.fun = fun
        self.position = position

    def __call__(self, *args: Any, **kwargs: Any) -> RetType:
        filename = args[self.position]
        with loop_until_access_time_greater_than_modification_time(filename):
            out = self.fun(*args, **kwargs)
        return out

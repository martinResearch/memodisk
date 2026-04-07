"""Module to memoize function results on disk with python code and data dependencies tracking."""

import base64
import binascii
import builtins
import contextlib
import datetime
import dis
import functools
import glob
import hashlib
import importlib.metadata
import inspect
import io
import json
import os
import pathlib
import pickle
import shutil
import stat
import subprocess
import sys
import tempfile
import threading
import time
import types
from collections.abc import Callable, Generator
from dataclasses import asdict, dataclass, field
from importlib import import_module
from os.path import exists
from sysconfig import get_path
from types import ModuleType
from typing import (
    IO,
    Any,
    Literal,
    Protocol,
    cast,
    overload,
)

ArgumentHashValue = bytes | str


def get_python_lib() -> str:
    """Get the path to the Python library."""
    return get_path("purelib")


# numpy as numba used to get the random states before and after function call
# could use plugin approach instead
numpy: ModuleType | None
try:
    numpy = __import__("numpy")
except ImportError:
    numpy = None


max_bytes = 2**31 - 1
disk_cache_dir = os.path.join(os.path.abspath(os.path.dirname(sys.argv[0])), "__memodisk__")
user_ignore_files: set[str] = set()  # should contain / not \
ignore_folders = [os.path.dirname(os.__file__), get_python_lib()]
strings_filter_out = {
    os.path.normpath(os.path.join("python", "debugpy", "_vendored", "pydevd")),
    "__pycache__",
}
compiled_library_extensions = frozenset({".pyd", ".so", ".dll", ".dylib"})
skip_check_global_changed = {"<class 'numba.core.registry.CPUDispatcher'>"}

# Pre-compute normalized path for this file (used to filter out memodisk internals)
_this_file = os.path.normpath(__file__)

# Tool ID for sys.monitoring (IDs 3-4 are freely available for tools)
MEMODISK_TOOL_ID = 3

# helps to check caching loading worked in tests
__last_cache_loading__: str | None = None


@dataclass(frozen=True)
class FuncKeyType:
    """Dataclass describing a function uniquely used as key in some dictionaries"""

    func_name: str
    filename: str
    line_number: int


def get_numba_dispatcher_hash(x: Any) -> str:
    return hashlib.sha256(x.__code__.co_code).hexdigest()


hashing_func_map: dict[Any, Callable[[Any], str]] = {
    str: str,
    int: str,
    types.FunctionType: lambda x: x.__qualname__,
    types.MethodType: lambda x: x.__qualname__,
    types.ModuleType: lambda x: f"{x.__name__}:{getattr(x, '__file__', 'builtin')}",
    "<class 'numba.core.registry.CPUDispatcher'>": get_numba_dispatcher_hash,
}


def get_last_cache_loading() -> str | None:
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
    module: str | None
    filename: str
    bytecode_hash: str
    global_vars: dict[str, str]
    closure_vars: dict[str, str]
    package_versions: dict[str, str] = field(default_factory=dict)
    compiled_dependencies: dict[str, str] = field(default_factory=dict)
    file_last_modified_date_str: str | None = None


@dataclass(frozen=True)
class FunctionDependencies:
    """Dataclass to store information on code and data dependency"""

    code: list
    data: list
    inherited: list
    random_states: dict | None
    external_processes: list = field(default_factory=list)
    ambient_time_sources: list[str] = field(default_factory=list)
    ambient_environment_sources: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ExternalProcessDependency:
    """Metadata for an observed external process dependency."""

    executable_path: str
    tracking_kind: str
    tracking_complete: bool


GlobalsType = dict[FuncKeyType, dict[str, str]]


@dataclass
class TraceScopeState:
    """Per-execution tracing state isolated to the current thread/scope."""

    code_dependencies_counters: dict[FuncKeyType, int] = field(default_factory=dict)
    function_qualified_name: dict[FuncKeyType, str] = field(default_factory=dict)
    function_bytecode_hash: dict[FuncKeyType, str] = field(default_factory=dict)
    function_modules: dict[FuncKeyType, str | None] = field(default_factory=dict)
    data_dependencies_counters: dict[DataDependency, int] = field(default_factory=dict)
    inherited_dependencies_counters: dict[DataDependency, int] = field(default_factory=dict)
    globals: GlobalsType = field(default_factory=dict)
    closure_vars: GlobalsType = field(default_factory=dict)
    package_versions: dict[FuncKeyType, dict[str, str]] = field(default_factory=dict)
    compiled_dependencies: dict[FuncKeyType, dict[str, str]] = field(default_factory=dict)
    external_processes: dict[str, ExternalProcessDependency] = field(default_factory=dict)
    ambient_time_sources: set[str] = field(default_factory=set)
    ambient_environment_sources: set[str] = field(default_factory=set)
    external_process_mode: str = "direct"
    global_changed: bool = False
    global_change_messages: list[str] = field(default_factory=list)
    has_dynamic_globals: bool = False
    dynamic_globals_functions: list[str] = field(default_factory=list)


builtins_open = builtins.open
io_open = io.open
pathlib_Path_open = pathlib.Path.open
subprocess_Popen = subprocess.Popen
os_system = os.system
open_delay = 0.001
lock_metadata_grace_seconds = 0.1
lock_owner_filename = "owner.json"
_OPEN_ATTRIBUTE = "open"
_POPEN_ATTRIBUTE = "Popen"
_SYSTEM_ATTRIBUTE = "system"


@dataclass(frozen=True)
class ResultSerializer[SerializedValue]:
    """Serialize memoized results to and from bytes."""

    name: str
    dumps: Callable[[SerializedValue], bytes]
    loads: Callable[[bytes], SerializedValue]


@dataclass(frozen=True)
class ArgumentHasher:
    """Compute custom cache-key material from call arguments."""

    name: str
    hash_args: Callable[[tuple[Any, ...], dict[str, Any]], ArgumentHashValue]


PICKLE_RESULT_SERIALIZER = ResultSerializer[Any](
    name="pickle",
    dumps=lambda value: pickle.dumps(value, protocol=4),
    loads=pickle.loads,
)


def _callable_signature(value: Any) -> tuple[str | None, str | None, Any | None]:
    return (
        getattr(value, "__module__", None),
        getattr(value, "__qualname__", None),
        getattr(value, "__self__", None),
    )


_AMBIENT_TIME_DIRECT_CALLS: dict[tuple[str | None, str | None, Any | None], str] = {
    _callable_signature(time.time): "time.time",
    _callable_signature(time.time_ns): "time.time_ns",
    _callable_signature(datetime.datetime.now): "datetime.datetime.now",
    (None, "datetime.utcnow", datetime.datetime): "datetime.datetime.utcnow",
    _callable_signature(datetime.datetime.today): "datetime.datetime.today",
    _callable_signature(datetime.date.today): "datetime.date.today",
}

_AMBIENT_ENVIRONMENT_DIRECT_CALLS: dict[tuple[str | None, str | None, Any | None], str] = {
    _callable_signature(os.getenv): "os.getenv",
    _callable_signature(os.getcwd): "os.getcwd",
    _callable_signature(pathlib.Path.cwd): "pathlib.Path.cwd",
}


def set_cache_dir(folder: str) -> None:
    global disk_cache_dir
    disk_cache_dir = folder


def _normalize_command_path(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, bytes):
        return os.fsdecode(value)
    if isinstance(value, os.PathLike):
        return os.fspath(value)
    if isinstance(value, str):
        return value
    return None


def _resolve_executable_path(command: Any) -> str | None:
    candidate = _normalize_command_path(command)
    if not candidate:
        return None

    if os.path.isabs(candidate) or os.path.dirname(candidate):
        normalized = os.path.abspath(os.path.normpath(candidate))
        if os.path.exists(normalized):
            return normalized
        return None

    resolved = shutil.which(candidate)
    if resolved is None:
        return None
    return os.path.abspath(os.path.normpath(resolved))


def _resolve_subprocess_executable(
    args: Any,
    executable: Any = None,
    shell: bool = False,
) -> str | None:
    if executable is not None:
        return _resolve_executable_path(executable)

    if shell:
        if os.name == "nt":
            shell_executable = os.environ.get("COMSPEC") or shutil.which("cmd.exe")
        else:
            shell_executable = "/bin/sh"
        return _resolve_executable_path(shell_executable)

    if isinstance(args, str | bytes | os.PathLike):
        return _resolve_executable_path(args)

    try:
        command_args = list(args)
    except TypeError:
        return None

    if not command_args:
        return None
    return _resolve_executable_path(command_args[0])


def memoized_popen_wrapper(*popenargs: Any, **kwargs: Any) -> Any:
    if popenargs:
        executable_path = _resolve_subprocess_executable(
            popenargs[0],
            executable=kwargs.get("executable"),
            shell=bool(kwargs.get("shell", False)),
        )
        if executable_path is not None:
            tracer.add_external_process_dependency(executable_path)

    return subprocess_Popen(*popenargs, **kwargs)


def memoized_system_wrapper(command: str) -> int:
    executable_path = _resolve_subprocess_executable(command, shell=True)
    if executable_path is not None:
        tracer.add_external_process_dependency(executable_path)
    return os_system(command)


def _write_big_bytes(data: bytes, file_path: str) -> None:
    with builtins_open(file_path, "wb") as f_out:
        for idx in range(0, len(data), max_bytes):
            f_out.write(data[idx : idx + max_bytes])


def _read_big_bytes(file_path: str) -> bytes:
    bytes_in = bytearray(0)
    input_size = os.path.getsize(file_path)
    with builtins_open(file_path, "rb") as f_in:
        for _ in range(0, input_size, max_bytes):
            bytes_in += f_in.read(max_bytes)

    return bytes(bytes_in)


def serialize_big_data[SerializedValue](
    data: SerializedValue,
    file_path: str,
    serializer: ResultSerializer[SerializedValue],
) -> None:
    _write_big_bytes(serializer.dumps(data), file_path)


def deserialize_big_data[SerializedValue](
    file_path: str,
    serializer: ResultSerializer[SerializedValue],
) -> SerializedValue:
    return serializer.loads(_read_big_bytes(file_path))


def pickle_big_data(data: Any, file_path: str) -> None:
    serialize_big_data(data, file_path, PICKLE_RESULT_SERIALIZER)


def unpickle_big_data(file_path: str) -> Any:
    return deserialize_big_data(file_path, PICKLE_RESULT_SERIALIZER)


def _normalize_argument_hash_value(value: ArgumentHashValue) -> bytes:
    if isinstance(value, bytes):
        return value
    if isinstance(value, str):
        return value.encode("utf-8")
    raise TypeError(f"ArgumentHasher.hash_args must return bytes or str, got {type(value).__qualname__}")


def load_cached_call_arguments(dependencies_file: str) -> dict[str, Any]:
    """Load stored args/kwargs for a cache entry from its dependencies JSON file."""
    with builtins_open(dependencies_file, "r") as fh:
        all_dependencies = json.load(fh)

    call_arguments_file = all_dependencies.get("call_arguments_file")
    if not call_arguments_file:
        raise ValueError(
            f"No stored call arguments found for {dependencies_file}. "
            "Cache entry was likely created without store_call_arguments=True."
        )
    if not os.path.exists(call_arguments_file):
        raise FileNotFoundError(call_arguments_file)

    payload = unpickle_big_data(call_arguments_file)
    assert isinstance(payload, dict)
    return payload


def get_globals_from_code(code: types.CodeType) -> list[str]:
    """Extract global variable names referenced by the code object.

    Uses dis.get_instructions() for stable bytecode introspection
    across Python 3.12+.
    """
    return sorted(
        {instr.argval for instr in dis.get_instructions(code) if instr.opname in ("LOAD_GLOBAL", "STORE_GLOBAL")}
    )


def _get_instruction_loaded_names(instr: dis.Instruction) -> tuple[str, ...]:
    argval = instr.argval
    if isinstance(argval, str):
        return (argval,)
    if isinstance(argval, tuple) and all(isinstance(name, str) for name in argval):
        return argval
    return ()


def _is_supported_load_instruction(instr: dis.Instruction) -> bool:
    return instr.opname in {"LOAD_GLOBAL", "LOAD_DEREF", "LOAD_NAME"} or instr.opname.startswith("LOAD_FAST")


def _get_frame_value(frame: types.FrameType, instr: dis.Instruction) -> Any:
    loaded_names = _get_instruction_loaded_names(instr)
    if not loaded_names:
        return None
    name = loaded_names[0]

    if instr.opname.startswith("LOAD_FAST"):
        return frame.f_locals.get(name)

    if instr.opname == "LOAD_GLOBAL":
        if name in frame.f_globals:
            return frame.f_globals[name]
        return frame.f_builtins.get(name)

    if instr.opname in ("LOAD_DEREF", "LOAD_NAME"):
        if name in frame.f_locals:
            return frame.f_locals[name]
        if name in frame.f_globals:
            return frame.f_globals[name]
        return frame.f_builtins.get(name)

    return None


def _get_direct_ambient_source_name(
    value: Any,
    known_sources: dict[tuple[str | None, str | None, Any | None], str],
) -> str | None:
    self_obj = getattr(value, "__self__", None)
    if self_obj is not None and not isinstance(self_obj, ModuleType | type):
        return None

    return known_sources.get(_callable_signature(value))


def _get_ambient_time_source_name(value: Any) -> str | None:
    return _get_direct_ambient_source_name(value, _AMBIENT_TIME_DIRECT_CALLS)


def _get_ambient_environment_source_name(value: Any) -> str | None:
    return _get_direct_ambient_source_name(value, _AMBIENT_ENVIRONMENT_DIRECT_CALLS)


def _next_call_index(instructions: list[dis.Instruction], start_index: int) -> int | None:
    ignored = {"CACHE", "EXTENDED_ARG", "NOP", "PUSH_NULL", "KW_NAMES", "PRECALL"}
    for idx in range(start_index, len(instructions)):
        opname = instructions[idx].opname
        if opname in ignored:
            continue
        if opname == "CALL":
            return idx
        return None
    return None


def _next_operation_kind(instructions: list[dis.Instruction], start_index: int) -> str | None:
    ignored = {
        "CACHE",
        "COPY",
        "EXTENDED_ARG",
        "FORMAT_VALUE",
        "KW_NAMES",
        "LOAD_ATTR",
        "LOAD_CONST",
        "LOAD_DEREF",
        "LOAD_GLOBAL",
        "LOAD_METHOD",
        "LOAD_NAME",
        "NOP",
        "PRECALL",
        "PUSH_NULL",
    }
    for idx in range(start_index, len(instructions)):
        opname = instructions[idx].opname
        if opname in ignored or opname.startswith("LOAD_FAST"):
            continue
        if opname == "CALL":
            return "call"
        if opname == "BINARY_SUBSCR":
            return "subscript"
        if opname == "BINARY_OP" and instructions[idx].argrepr == "[]":
            return "subscript"
        return None
    return None


def _next_instruction_index(
    instructions: list[dis.Instruction], start_index: int, ignored: set[str] | None = None
) -> int | None:
    ignored_opnames = {"CACHE", "EXTENDED_ARG", "NOP"}
    if ignored is not None:
        ignored_opnames |= ignored

    for idx in range(start_index, len(instructions)):
        if instructions[idx].opname in ignored_opnames:
            continue
        return idx
    return None


def _ambient_environment_alias_kind(value: Any, attrs: tuple[str, ...]) -> str | None:
    if value is os:
        if attrs == ("environ",):
            return "os.environ"
        if attrs == ("getenv",):
            return "os.getenv"
        if attrs == ("getcwd",):
            return "os.getcwd"
    if value is os.environ and not attrs:
        return "os.environ"
    if value is pathlib:
        if attrs == ("Path", "cwd"):
            return "pathlib.Path.cwd"
    if value is pathlib.Path and attrs == ("cwd",):
        return "pathlib.Path.cwd"
    return None


def _detect_ambient_time_sources(code: types.CodeType, frame: types.FrameType) -> set[str]:
    instructions = list(dis.get_instructions(code))
    sources: set[str] = set()

    for idx, instr in enumerate(instructions):
        if not _is_supported_load_instruction(instr):
            continue

        value = _get_frame_value(frame, instr)
        direct_source = _get_ambient_time_source_name(value)
        if direct_source is not None and _next_call_index(instructions, idx + 1) is not None:
            sources.add(direct_source)
            continue

        attrs: list[str] = []
        scan_idx = idx + 1
        while scan_idx < len(instructions) and instructions[scan_idx].opname in {"LOAD_ATTR", "LOAD_METHOD"}:
            attr_name = instructions[scan_idx].argval
            if isinstance(attr_name, str):
                attrs.append(attr_name)
            scan_idx += 1

        if not attrs or _next_call_index(instructions, scan_idx) is None:
            continue

        if value is time and tuple(attrs) in {("time",), ("time_ns",)}:
            sources.add(f"time.{attrs[0]}")
            continue

        if value is datetime:
            if tuple(attrs) == ("datetime", "now"):
                sources.add("datetime.datetime.now")
            elif tuple(attrs) == ("datetime", "utcnow"):
                sources.add("datetime.datetime.utcnow")
            elif tuple(attrs) == ("datetime", "today"):
                sources.add("datetime.datetime.today")
            elif tuple(attrs) == ("date", "today"):
                sources.add("datetime.date.today")
            continue

        if value is datetime.datetime and tuple(attrs) in {("now",), ("utcnow",), ("today",)}:
            sources.add(f"datetime.datetime.{attrs[0]}")
            continue

        if value is datetime.date and tuple(attrs) == ("today",):
            sources.add("datetime.date.today")

    return sources


def _detect_ambient_environment_sources(code: types.CodeType, frame: types.FrameType) -> set[str]:
    instructions = list(dis.get_instructions(code))
    sources: set[str] = set()
    local_aliases: dict[str, str] = {}

    for idx, instr in enumerate(instructions):
        if not _is_supported_load_instruction(instr):
            continue

        value = _get_frame_value(frame, instr)
        loaded_names = _get_instruction_loaded_names(instr)
        primary_loaded_name = loaded_names[0] if loaded_names else None
        if instr.opname.startswith("LOAD_FAST") and primary_loaded_name in local_aliases:
            value = local_aliases[primary_loaded_name]

        direct_source = _get_ambient_environment_source_name(value)
        if direct_source is not None and _next_operation_kind(instructions, idx + 1) == "call":
            sources.add(direct_source)
            continue

        attrs: list[str] = []
        scan_idx = idx + 1
        while scan_idx < len(instructions) and instructions[scan_idx].opname in {"LOAD_ATTR", "LOAD_METHOD"}:
            attr_name = instructions[scan_idx].argval
            if isinstance(attr_name, str):
                attrs.append(attr_name)
            scan_idx += 1

        alias_kind = None
        if not isinstance(value, str):
            alias_kind = _ambient_environment_alias_kind(value, tuple(attrs))
        next_idx = _next_instruction_index(instructions, scan_idx)
        if alias_kind is not None and next_idx is not None:
            next_instr = instructions[next_idx]
            if next_instr.opname == "STORE_FAST" and isinstance(next_instr.argval, str):
                local_aliases[next_instr.argval] = alias_kind

        operation_kind = _next_operation_kind(instructions, scan_idx)
        if operation_kind is None:
            continue

        if isinstance(value, str) and value == "os.environ":
            if not attrs and operation_kind == "subscript":
                sources.add("os.environ.__getitem__")
            elif tuple(attrs) == ("get",) and operation_kind == "call":
                sources.add("os.environ.get")
            continue

        if isinstance(value, str) and value in {"os.getenv", "os.getcwd", "pathlib.Path.cwd"}:
            if not attrs and operation_kind == "call":
                sources.add(value)
            continue

        if value is os:
            if tuple(attrs) == ("getenv",) and operation_kind == "call":
                sources.add("os.getenv")
            elif tuple(attrs) == ("getcwd",) and operation_kind == "call":
                sources.add("os.getcwd")
            elif tuple(attrs) == ("environ",) and operation_kind == "subscript":
                sources.add("os.environ.__getitem__")
            elif tuple(attrs) == ("environ", "get") and operation_kind == "call":
                sources.add("os.environ.get")
            continue

        if value is os.environ:
            if not attrs and operation_kind == "subscript":
                sources.add("os.environ.__getitem__")
            elif tuple(attrs) == ("get",) and operation_kind == "call":
                sources.add("os.environ.get")
            continue

        if value is pathlib:
            if tuple(attrs) == ("Path", "cwd") and operation_kind == "call":
                sources.add("pathlib.Path.cwd")
            continue

        if value is pathlib.Path and tuple(attrs) == ("cwd",) and operation_kind == "call":
            sources.add("pathlib.Path.cwd")

    return sources


def _lock_owner_file_path(lock_path: str) -> str:
    return os.path.join(lock_path, lock_owner_filename)


def _is_pid_running(pid: int) -> bool:
    if pid <= 0:
        return False

    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        process_query_limited_information = 0x1000
        still_active = 259

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        handle = kernel32.OpenProcess(process_query_limited_information, False, pid)
        if not handle:
            return False

        try:
            exit_code = wintypes.DWORD()
            if not kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
                return False
            return exit_code.value == still_active
        finally:
            kernel32.CloseHandle(handle)

    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def _write_lock_owner_metadata(lock_path: str) -> None:
    owner_path = _lock_owner_file_path(lock_path)
    with builtins_open(owner_path, "w", encoding="utf-8") as fh:
        json.dump({"pid": os.getpid(), "created_at": time.time()}, fh)


def _lock_metadata_is_stale(lock_path: str) -> bool:
    owner_path = _lock_owner_file_path(lock_path)
    try:
        with builtins_open(owner_path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
    except FileNotFoundError:
        try:
            return time.time() - os.path.getmtime(lock_path) >= lock_metadata_grace_seconds
        except FileNotFoundError:
            return False
    except (json.JSONDecodeError, OSError, ValueError, TypeError):
        return True

    pid = payload.get("pid")
    if isinstance(pid, int):
        return not _is_pid_running(pid)
    return True


def _remove_stale_lock_directory(lock_path: str) -> bool:
    owner_path = _lock_owner_file_path(lock_path)
    try:
        with contextlib.suppress(FileNotFoundError):
            os.remove(owner_path)
        os.rmdir(lock_path)
        return True
    except OSError:
        return False


def _remove_path_with_retry(path: str, remove_dir: bool = False) -> None:
    for _ in range(1000):
        try:
            if remove_dir:
                os.rmdir(path)
            else:
                os.remove(path)
            return
        except FileNotFoundError:
            return
        except PermissionError:
            time.sleep(open_delay)
        except OSError:
            return


_cache_process_lock_state = threading.local()


def _get_process_lock_depths() -> dict[str, int]:
    depths = getattr(_cache_process_lock_state, "depths", None)
    if depths is None:
        depths = {}
        _cache_process_lock_state.depths = depths
    return depths


@contextlib.contextmanager
def _cache_process_lock(cache_prefix: str) -> Generator[None, None, None]:
    depths = _get_process_lock_depths()
    current_depth = depths.get(cache_prefix, 0)
    if current_depth > 0:
        depths[cache_prefix] = current_depth + 1
        try:
            yield None
        finally:
            if depths[cache_prefix] == 1:
                depths.pop(cache_prefix, None)
            else:
                depths[cache_prefix] -= 1
        return

    lock_path = f"{cache_prefix}.lock"
    os.makedirs(os.path.dirname(lock_path), exist_ok=True)
    while True:
        try:
            os.mkdir(lock_path)
            try:
                _write_lock_owner_metadata(lock_path)
            except Exception:
                with contextlib.suppress(OSError):
                    _remove_stale_lock_directory(lock_path)
                raise
            break
        except FileExistsError:
            if _lock_metadata_is_stale(lock_path) and _remove_stale_lock_directory(lock_path):
                continue
            time.sleep(open_delay)

    depths[cache_prefix] = 1
    try:
        yield None
    finally:
        depths.pop(cache_prefix, None)
        _remove_path_with_retry(_lock_owner_file_path(lock_path))
        _remove_path_with_retry(lock_path, remove_dir=True)


def _write_bytes_atomic(data: bytes, file_path: str) -> None:
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    tmp_path = ""
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            delete=False,
            dir=os.path.dirname(file_path),
            prefix=f".{os.path.basename(file_path)}.",
            suffix=".tmp",
        ) as tmp_file:
            tmp_path = tmp_file.name
        _write_big_bytes(data, tmp_path)
        os.replace(tmp_path, file_path)
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


def _serialize_big_data_atomic[SerializedValue](
    data: SerializedValue,
    file_path: str,
    serializer: ResultSerializer[SerializedValue],
) -> None:
    _write_bytes_atomic(serializer.dumps(data), file_path)


def _write_json_atomic(file_path: str, payload: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    tmp_path = ""
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            delete=False,
            dir=os.path.dirname(file_path),
            prefix=f".{os.path.basename(file_path)}.",
            suffix=".tmp",
            encoding="utf-8",
        ) as tmp_file:
            tmp_path = tmp_file.name
            json.dump(payload, tmp_file, indent=4)
        os.replace(tmp_path, file_path)
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


def _uses_dynamic_globals(code: types.CodeType) -> bool:
    """Detect if a code object calls globals() or locals().

    These bypass LOAD_GLOBAL and make dependency tracking incomplete.
    """
    for instr in dis.get_instructions(code):
        if instr.opname == "LOAD_GLOBAL" and instr.argval in ("globals", "locals"):
            return True
    return False


def _canonical_hash(obj: Any) -> bytes:
    """Produce a canonical pickle-based hash for arguments.

    Normalizes dicts (sort keys) and sets/frozensets (sort elements)
    so that insertion order doesn't affect the hash.
    """
    return pickle.dumps(_canonicalize(obj), protocol=4)


def _canonicalize(obj: Any) -> Any:
    """Recursively normalize an object for deterministic pickling."""
    if isinstance(obj, dict):
        items = [(_canonicalize(k), _canonicalize(v)) for k, v in obj.items()]
        items.sort(key=lambda item: pickle.dumps(item[0], protocol=4))
        return ("dict", tuple(items))
    if isinstance(obj, set | frozenset):
        values = [_canonicalize(v) for v in obj]
        values.sort(key=lambda value: pickle.dumps(value, protocol=4))
        return (type(obj).__name__, tuple(values))
    if isinstance(obj, list):
        return ("list", tuple(_canonicalize(v) for v in obj))
    if isinstance(obj, tuple):
        return ("tuple", tuple(_canonicalize(v) for v in obj))
    return obj


def get_global_hash(
    name: str,
    variable: Any,
    frame: types.FrameType | None,
    co: types.CodeType | None,
) -> str:
    variable_type = type(variable)
    if str(variable_type) in hashing_func_map:
        hash_str = hashing_func_map[str(variable_type)](variable)
    elif variable_type in hashing_func_map:
        hash_str = hashing_func_map[variable_type](variable)
    else:
        try:
            pickled_var = pickle.dumps(variable)
        except (pickle.PicklingError, TypeError, AttributeError) as e:
            if frame is None:
                raise RuntimeError(f"Could not pickle global variable {name}") from e
            else:
                assert co is not None
                raise RuntimeError(
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


def _get_runtime_hash(value_name: str, value: Any) -> str:
    try:
        return get_global_hash(value_name, value, None, None)
    except RuntimeError:
        return f"unpicklable:{type(value).__qualname__}:{id(value)}"


def _get_file_last_modified_date_str(file_path: str) -> str | None:
    normalized_path = os.path.normpath(file_path)
    if not os.path.exists(normalized_path):
        return None
    return str(datetime.datetime.fromtimestamp(os.stat(normalized_path).st_mtime))


@functools.lru_cache(maxsize=1)
def _get_packages_distributions_map() -> dict[str, list[str]]:
    return dict(importlib.metadata.packages_distributions())


@functools.cache
def _get_installed_package_version(package_name: str) -> str | None:
    try:
        return importlib.metadata.version(package_name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _is_site_packages_path(file_path: str | None) -> bool:
    if not file_path:
        return False

    normalized_path = os.path.abspath(os.path.normpath(file_path))
    purelib_path = os.path.abspath(os.path.normpath(get_python_lib()))
    try:
        return os.path.commonpath([normalized_path, purelib_path]) == purelib_path
    except ValueError:
        return False


def _is_compiled_library_path(file_path: str | None) -> bool:
    if not file_path:
        return False
    return os.path.splitext(file_path)[1].lower() in compiled_library_extensions


def _get_package_versions_for_module(module: ModuleType | None) -> dict[str, str]:
    if module is None:
        return {}

    module_name = getattr(module, "__name__", None)
    module_file = getattr(module, "__file__", None)
    if module_name is None or not _is_site_packages_path(module_file):
        return {}

    top_level_name = module_name.partition(".")[0]
    package_versions: dict[str, str] = {}
    for package_name in _get_packages_distributions_map().get(top_level_name, []):
        version = _get_installed_package_version(package_name)
        if version is not None:
            package_versions[package_name] = version
    return package_versions


def _get_compiled_dependencies_for_module(module: ModuleType | None) -> dict[str, str]:
    if module is None:
        return {}

    module_name = getattr(module, "__name__", None)
    module_file = getattr(module, "__file__", None)
    if module_name is None or not _is_site_packages_path(module_file):
        return {}

    top_level_name = module_name.partition(".")[0]
    compiled_dependencies: dict[str, str] = {}
    for loaded_name, loaded_module in tuple(sys.modules.items()):
        if not loaded_name.startswith(top_level_name):
            continue

        loaded_file = getattr(loaded_module, "__file__", None)
        if not _is_compiled_library_path(loaded_file) or not _is_site_packages_path(loaded_file):
            continue
        if not isinstance(loaded_file, str | bytes | os.PathLike):
            continue

        normalized_file = os.path.abspath(os.path.normpath(str(loaded_file)))
        last_modified_date_str = _get_file_last_modified_date_str(normalized_file)
        if last_modified_date_str is not None:
            compiled_dependencies[normalized_file] = last_modified_date_str
    return compiled_dependencies


def _get_package_versions_for_value(value: Any) -> dict[str, str]:
    if isinstance(value, ModuleType):
        return _get_package_versions_for_module(value)

    module = inspect.getmodule(value)
    if module is None:
        module_name = getattr(value, "__module__", None)
        if module_name is not None:
            with contextlib.suppress(Exception):
                module = import_module(module_name)

    return _get_package_versions_for_module(module)


def _get_compiled_dependencies_for_value(value: Any) -> dict[str, str]:
    if isinstance(value, ModuleType):
        return _get_compiled_dependencies_for_module(value)

    module = inspect.getmodule(value)
    if module is None:
        module_name = getattr(value, "__module__", None)
        if module_name is not None:
            with contextlib.suppress(Exception):
                module = import_module(module_name)

    return _get_compiled_dependencies_for_module(module)


def _merge_package_versions(target: dict[str, str], package_versions: dict[str, str]) -> None:
    for package_name, version in package_versions.items():
        target[package_name] = version


def _merge_file_dependencies(target: dict[str, str], file_dependencies: dict[str, str]) -> None:
    for file_path, last_modified_date_str in file_dependencies.items():
        target[file_path] = last_modified_date_str


class CodeBackedCallable(Protocol):
    __code__: types.CodeType
    __qualname__: str
    __module__: str | None  # type: ignore[assignment]
    __globals__: dict[str, Any]

    def __call__(self, *args: Any, **kwargs: Any) -> Any: ...


def _as_code_backed_callable(value: object) -> CodeBackedCallable | None:
    if all(hasattr(value, attr) for attr in ("__code__", "__qualname__", "__module__", "__globals__")):
        return cast(CodeBackedCallable, value)
    return None


def _require_code_backed_callable(func: Callable[..., Any]) -> CodeBackedCallable:
    code_backed = _as_code_backed_callable(func)
    if code_backed is None:
        raise TypeError(f"Memoize requires a Python function object, got {type(func).__qualname__}")
    return code_backed


def _get_callable_dependency(func: CodeBackedCallable) -> CodeDependency:
    code = func.__code__
    globals_dict = getattr(func, "__globals__", {})
    builtins_dict = builtins.__dict__

    global_vars: dict[str, str] = {}
    package_versions: dict[str, str] = {}
    compiled_dependencies: dict[str, str] = {}
    for name in get_globals_from_code(code):
        if name in builtins_dict or name not in globals_dict:
            continue
        value = globals_dict[name]
        if isinstance(value, type | types.BuiltinFunctionType):
            continue
        global_vars[name] = _get_runtime_hash(name, value)
        _merge_package_versions(package_versions, _get_package_versions_for_value(value))
        _merge_file_dependencies(compiled_dependencies, _get_compiled_dependencies_for_value(value))

    closure_vars: dict[str, str] = {}
    closure_nonlocals = inspect.getclosurevars(func).nonlocals
    for name in code.co_freevars:
        if name not in closure_nonlocals:
            continue
        value = closure_nonlocals[name]
        if isinstance(value, type | types.BuiltinFunctionType):
            continue
        closure_vars[name] = _get_runtime_hash(name, value)
        _merge_package_versions(package_versions, _get_package_versions_for_value(value))
        _merge_file_dependencies(compiled_dependencies, _get_compiled_dependencies_for_value(value))

    _merge_package_versions(package_versions, _get_package_versions_for_module(inspect.getmodule(func)))
    _merge_file_dependencies(compiled_dependencies, _get_compiled_dependencies_for_module(inspect.getmodule(func)))

    return CodeDependency(
        function_qualified_name=func.__qualname__,
        module=func.__module__,
        filename=os.path.normpath(code.co_filename),
        bytecode_hash=get_bytecode_hash(code),
        global_vars=global_vars,
        closure_vars=closure_vars,
        package_versions=package_versions,
        compiled_dependencies=compiled_dependencies,
        file_last_modified_date_str=_get_file_last_modified_date_str(code.co_filename),
    )


def _unwrap_memoized_callable(func: Callable[..., Any]) -> Callable[..., Any]:
    code_backed = _as_code_backed_callable(func)
    if code_backed is not None and code_backed.__code__.co_name == "_memoize_wrapper":
        wrapped = getattr(func, "__wrapped__", None)
        if callable(wrapped):
            return wrapped
    return func


def _callable_matches_dependency(func: object, entry_code: CodeDependency) -> bool:
    code_backed = _as_code_backed_callable(func)
    if code_backed is None:
        return False
    code = code_backed.__code__
    if os.path.normpath(code.co_filename) != entry_code.filename:
        return False

    qualnames = {
        code_backed.__qualname__,
        getattr(code, "co_qualname", None),
    }
    return entry_code.function_qualified_name in qualnames


def _resolve_dependency_callable(func: Callable[..., Any], entry_code: CodeDependency) -> CodeBackedCallable | None:
    for wrapped_func in _iter_wrapped_callables(func):
        if _callable_matches_dependency(wrapped_func, entry_code):
            return _as_code_backed_callable(wrapped_func)
        candidate = _unwrap_memoized_callable(wrapped_func)
        if _callable_matches_dependency(candidate, entry_code):
            return _as_code_backed_callable(candidate)

    node: ModuleType | None | Callable
    if entry_code.module not in ("__main__", None):
        assert entry_code.module is not None
        node = import_module(entry_code.module)
    else:
        node = inspect.getmodule(func)

    for name in entry_code.function_qualified_name.split("."):
        new_node = getattr(node, name, None)
        if new_node is None:
            return None
        node = new_node

    if isinstance(node, property):
        node = node.fget
    if not callable(node):
        return None

    if _callable_matches_dependency(node, entry_code):
        return _as_code_backed_callable(node)

    candidate = _unwrap_memoized_callable(cast(Callable[..., Any], node))
    if _callable_matches_dependency(candidate, entry_code):
        return _as_code_backed_callable(candidate)
    return None


def _iter_wrapped_callables(func: Callable[..., Any]) -> Generator[Callable[..., Any], None, None]:
    seen: set[int] = set()
    current: Callable[..., Any] | None = func
    while current is not None and id(current) not in seen:
        yield current
        seen.add(id(current))
        wrapped = getattr(current, "__wrapped__", None)
        current = wrapped if callable(wrapped) else None


def _install_runtime_wrappers() -> None:
    setattr(builtins, _OPEN_ATTRIBUTE, memoized_open_wrapper)
    setattr(io, _OPEN_ATTRIBUTE, memoized_open_wrapper)
    setattr(pathlib.Path, _OPEN_ATTRIBUTE, memoized_path_open_wrapper)
    setattr(subprocess, _POPEN_ATTRIBUTE, memoized_popen_wrapper)
    setattr(os, _SYSTEM_ATTRIBUTE, memoized_system_wrapper)


def _restore_runtime_wrappers() -> None:
    setattr(builtins, _OPEN_ATTRIBUTE, builtins_open)
    setattr(io, _OPEN_ATTRIBUTE, io_open)
    setattr(pathlib.Path, _OPEN_ATTRIBUTE, pathlib_Path_open)
    setattr(subprocess, _POPEN_ATTRIBUTE, subprocess_Popen)
    setattr(os, _SYSTEM_ATTRIBUTE, os_system)


class Tracer:
    """Class used to track all dependencies of the functions.
    This is intended to be used as a singleton. A single class instance will track dependencies for all
    memoized functions.
    This is to avoid cascading callbacks when calling the trace function when we have nested memoized functions.
    This instance counts how many time a function is called and a dta file accessed.
    To get the dependencies of a function we then need to compare the counters before and after the function execution

    Uses sys.monitoring (PEP 669) with PY_START events for dependency tracking.
    This replaces the old sys.setprofile approach and provides:
    - Per-code-object callbacks with DISABLE returns for zero-overhead filtering
    - No conflicts with debuggers/profilers (separate tool IDs)
    - No BaseException hacks for global change detection
    """

    def __init__(self) -> None:
        self.is_registered = False
        self._registration_lock = threading.RLock()
        self._thread_state = threading.local()
        self._active_scope_count = 0
        self._ignore_files = [
            _this_file,
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
        self._thread_state.scope_stack = []

    def _get_scope_stack(self) -> list[TraceScopeState]:
        stack = getattr(self._thread_state, "scope_stack", None)
        if stack is None:
            stack = []
            self._thread_state.scope_stack = stack
        return stack

    def begin_scope(self) -> TraceScopeState:
        scope = TraceScopeState()
        stack = self._get_scope_stack()
        stack.append(scope)

        with self._registration_lock:
            if self._active_scope_count == 0:
                self.register()
                _install_runtime_wrappers()
            self._active_scope_count += 1

        return scope

    def end_scope(self, scope: TraceScopeState) -> None:
        stack = self._get_scope_stack()
        assert stack and stack[-1] is scope
        stack.pop()

        with self._registration_lock:
            self._active_scope_count -= 1
            if self._active_scope_count == 0:
                _restore_runtime_wrappers()
                self.unregister()

    def _get_active_scopes(self) -> list[TraceScopeState]:
        return list(self._get_scope_stack())

    def register(self) -> None:
        assert not self.is_registered
        sys.monitoring.use_tool_id(MEMODISK_TOOL_ID, "memodisk")
        sys.monitoring.set_events(MEMODISK_TOOL_ID, sys.monitoring.events.PY_START)
        sys.monitoring.register_callback(MEMODISK_TOOL_ID, sys.monitoring.events.PY_START, self._on_py_start)
        self.is_registered = True

    def unregister(self) -> None:
        assert self.is_registered
        sys.monitoring.set_events(MEMODISK_TOOL_ID, 0)
        sys.monitoring.register_callback(MEMODISK_TOOL_ID, sys.monitoring.events.PY_START, None)
        sys.monitoring.free_tool_id(MEMODISK_TOOL_ID)
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
        for scope in self._get_active_scopes():
            if dep in scope.data_dependencies_counters:
                scope.data_dependencies_counters[dep] += 1
            else:
                scope.data_dependencies_counters[dep] = 1

    def add_external_process_dependency(self, executable_path: str) -> None:
        normalized_path = os.path.abspath(os.path.normpath(executable_path))
        self.add_data_dependency(normalized_path)
        for scope in self._get_active_scopes():
            if scope.external_process_mode == "manual":
                continue
            scope.external_processes[normalized_path] = ExternalProcessDependency(
                executable_path=normalized_path,
                tracking_kind="direct",
                tracking_complete=False,
            )

    def add_inherited_dependency(self, dependency: DataDependency) -> None:
        for scope in self._get_active_scopes():
            if dependency in scope.inherited_dependencies_counters:
                scope.inherited_dependencies_counters[dependency] += 1
            else:
                scope.inherited_dependencies_counters[dependency] = 1

    @staticmethod
    def _get_module_name(frame: types.FrameType) -> str | None:
        module_name = frame.f_globals.get("__name__")
        if isinstance(module_name, str):
            return module_name
        module = inspect.getmodule(frame)
        if module is not None:
            return module.__name__
        return None

    _SKIP_TYPES = (type, types.BuiltinFunctionType)
    _SKIP_NAMES = frozenset(("<genexpr>", "<listcomp>", "<module>"))

    def _should_ignore(self, filename: str, function_name: str) -> bool:
        """Return True if this code object should be permanently ignored."""
        if filename == _this_file or filename.startswith("<"):
            return True
        if any(filename.lower().startswith(f.lower()) for f in ignore_folders):
            return True
        if filename in self._ignore_files or filename in user_ignore_files:
            return True
        if function_name in self._SKIP_NAMES:
            return True
        return any(s in filename for s in strings_filter_out)

    @staticmethod
    def _safe_hash(name: str, variable: Any, frame: types.FrameType, code: types.CodeType) -> str:
        """Hash a variable, falling back to type+id for unpicklable objects."""
        try:
            return get_global_hash(name, variable, frame, code)
        except (pickle.PicklingError, TypeError, AttributeError, RuntimeError):
            return f"unpicklable:{type(variable).__qualname__}:{id(variable)}"

    def _on_py_start(self, code: types.CodeType, instruction_offset: int) -> object:
        """PY_START callback for sys.monitoring.

        Called at the start of every Python function execution. Returns
        sys.monitoring.DISABLE for code objects we don't want to track.
        """
        active_scopes = self._get_active_scopes()
        if not active_scopes:
            return None

        filename = os.path.normpath(code.co_filename)
        if self._should_ignore(filename, code.co_name):
            return sys.monitoring.DISABLE

        func_key = FuncKeyType(
            func_name=code.co_name,
            filename=filename,
            line_number=code.co_firstlineno,
        )

        frame = sys._getframe(1)

        for scope in active_scopes:
            if func_key not in scope.function_bytecode_hash:
                scope.function_bytecode_hash[func_key] = get_bytecode_hash(code)

            if func_key not in scope.globals:
                # First visit: record global and closure variable hashes
                scope.globals[func_key] = {}
                scope.closure_vars[func_key] = {}
                scope.package_versions[func_key] = {}
                scope.compiled_dependencies[func_key] = {}

                for name in get_globals_from_code(code):
                    if name in frame.f_builtins or name not in frame.f_globals:
                        continue
                    variable = frame.f_globals[name]
                    if not isinstance(variable, self._SKIP_TYPES):
                        scope.globals[func_key][name] = self._safe_hash(name, variable, frame, code)
                        _merge_package_versions(
                            scope.package_versions[func_key], _get_package_versions_for_value(variable)
                        )
                        _merge_file_dependencies(
                            scope.compiled_dependencies[func_key], _get_compiled_dependencies_for_value(variable)
                        )

                for name in code.co_freevars:
                    if name in frame.f_builtins or name not in frame.f_locals:
                        continue
                    variable = frame.f_locals[name]
                    if not isinstance(variable, self._SKIP_TYPES):
                        scope.closure_vars[func_key][name] = self._safe_hash(name, variable, frame, code)
                        _merge_package_versions(
                            scope.package_versions[func_key], _get_package_versions_for_value(variable)
                        )
                        _merge_file_dependencies(
                            scope.compiled_dependencies[func_key], _get_compiled_dependencies_for_value(variable)
                        )

                module_name = self._get_module_name(frame)
                module = sys.modules.get(module_name) if module_name is not None else None
                _merge_package_versions(scope.package_versions[func_key], _get_package_versions_for_module(module))
                _merge_file_dependencies(
                    scope.compiled_dependencies[func_key], _get_compiled_dependencies_for_module(module)
                )
            else:
                # Subsequent visit: check globals haven't changed
                for name, prev_hash in scope.globals[func_key].items():
                    if name not in frame.f_globals:
                        continue
                    variable = frame.f_globals[name]
                    if str(type(variable)) in skip_check_global_changed:
                        continue
                    new_hash = self._safe_hash(name, variable, frame, code)
                    if new_hash != prev_hash:
                        scope.global_changed = True
                        scope.global_change_messages.append(
                            f'Global variable {name} used in function {code.co_name} in "{filename}",'
                            f" line {code.co_firstlineno} changed during two calls."
                        )

            if func_key not in scope.function_qualified_name:
                scope.function_qualified_name[func_key] = code.co_qualname

            if func_key not in scope.function_modules:
                scope.function_modules[func_key] = self._get_module_name(frame)

            scope.code_dependencies_counters[func_key] = scope.code_dependencies_counters.get(func_key, 0) + 1

            if _uses_dynamic_globals(code):
                scope.has_dynamic_globals = True
                scope.dynamic_globals_functions.append(f"{code.co_qualname} in {filename}")

            scope.ambient_time_sources.update(_detect_ambient_time_sources(code, frame))
            scope.ambient_environment_sources.update(_detect_ambient_environment_sources(code, frame))

        return None


tracer = Tracer()

_cache_key_locks_guard = threading.Lock()
_cache_key_locks: dict[str, Any] = {}
_CACHE_MISS = object()


def _get_cache_key_lock(cache_prefix: str) -> threading.RLock:
    with _cache_key_locks_guard:
        lock = _cache_key_locks.get(cache_prefix)
        if lock is None:
            lock = threading.RLock()
            _cache_key_locks[cache_prefix] = lock
        return lock


def _load_matching_cache_entry(
    func: Callable,
    cache_prefix: str,
    active_serializer: ResultSerializer[Any],
    mode: str,
    external_process_mode: str,
) -> tuple[Any, dict[str, Any] | None, str | None]:
    dep_pattern = f"{cache_prefix}_*_dependencies.json"
    for dep_candidate in sorted(glob.glob(dep_pattern)):
        try:
            with builtins_open(dep_candidate, "r") as fh:
                all_dependencies = json.load(fh)
        except (json.JSONDecodeError, OSError):
            continue
        if dependency_changed(func, all_dependencies):
            continue
        if _external_process_tracking_incomplete(all_dependencies) and (
            external_process_mode == "strict" or mode in ("safe", "strict")
        ):
            continue
        if _ambient_time_tracking_incomplete(all_dependencies) and mode in ("safe", "strict"):
            continue
        if _ambient_environment_tracking_incomplete(all_dependencies) and mode in ("safe", "strict"):
            continue

        result_candidate = dep_candidate.replace("_dependencies.json", "_result.pkl")
        if not exists(result_candidate):
            continue

        try:
            result = deserialize_big_data(result_candidate, active_serializer)
        except Exception:
            print(
                f"WARNING: Could not unpickle the data from {os.path.split(result_candidate)[1]},"
                " probably due to code or serializer change."
            )
            continue

        return result, all_dependencies, dep_candidate

    return _CACHE_MISS, None, None


def print_change(message: str) -> None:
    print("*****************")
    print("Memoization miss:")
    print(message)
    print("*****************")


def _external_process_tracking_incomplete(all_dependencies: dict[str, Any]) -> bool:
    return any(not entry.get("tracking_complete", False) for entry in all_dependencies.get("external_processes", []))


def _ambient_time_tracking_incomplete(all_dependencies: dict[str, Any]) -> bool:
    return bool(all_dependencies.get("ambient_time_sources", []))


def _ambient_environment_tracking_incomplete(all_dependencies: dict[str, Any]) -> bool:
    return bool(all_dependencies.get("ambient_environment_sources", []))


def dependency_changed(func: Callable, all_dependencies: dict) -> bool:
    """Detect if any of the dependencies of the function has changed."""
    ignore_code_changes = all_dependencies.get("ignore_code_changes", False)

    # Check random state
    if all_dependencies["random_states"] is not None:
        assert numpy is not None, "You need the numpy module"
        current_state = _get_numpy_random_state_b64()
        if current_state != all_dependencies["random_states"]["before"]["numpy"]:
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

        if any(s in filename for s in strings_filter_out):
            print(f"Dependencies of file {filename} filtered out")
            continue
        bytecode_hash = entry_code.bytecode_hash
        global_variables = entry_code.global_vars
        closure_variables = entry_code.closure_vars
        if not os.path.exists(filename):
            print_change(f"Could not find file {filename}.")
            return True

        for package_name, expected_version in entry_code.package_versions.items():
            current_version = _get_installed_package_version(package_name)
            if current_version != expected_version:
                print_change(
                    f"Package dependency {package_name} has been modified. Was {expected_version} now {current_version}"
                )
                return True

        for file_path, expected_last_modified_date_str in entry_code.compiled_dependencies.items():
            current_last_modified_date_str = _get_file_last_modified_date_str(file_path)
            if current_last_modified_date_str != expected_last_modified_date_str:
                print_change(
                    f"Compiled dependency {file_path} has been modified. "
                    f"Was {expected_last_modified_date_str} now {current_last_modified_date_str}"
                )
                return True

        current_file_last_modified_date_str = _get_file_last_modified_date_str(filename)
        file_mtime_unchanged = (
            entry_code.file_last_modified_date_str is not None
            and entry_code.file_last_modified_date_str == current_file_last_modified_date_str
        )

        # Retrieve the dependency function
        func_dep = _resolve_dependency_callable(func, entry_code)
        if func_dep is None:
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
            hash_str = _get_runtime_hash(name, variable)
            if hash_str != value:
                print_change(f"Global variable {name} used in {function_qualified_name} has been modified")
                return True

        # check closure variables did not change

        for name, value in closure_variables.items():
            dep_closure_vars = inspect.getclosurevars(func_dep).nonlocals
            variable = dep_closure_vars[name]
            hash_str = _get_runtime_hash(name, variable)
            if hash_str != value:
                print_change(f"Closure variable {name} used in {function_qualified_name} has been modified")
                return True
        if not ignore_code_changes and not file_mtime_unchanged:
            new_bytecode_hash = get_bytecode_hash(func_dep.__code__)
            if new_bytecode_hash != bytecode_hash:
                print_change(
                    f"Code dependency {function_qualified_name} has been modified."
                    f" Was {bytecode_hash} now {new_bytecode_hash}"
                )
                return True
    return False


def _get_numpy_random_state_b64() -> str | None:
    if numpy is None:
        return None
    return base64.b64encode(pickle.dumps(numpy.random.get_state())).decode("utf-8")


def get_dependencies_runtime(
    func: Callable[..., Any], *args: Any, external_process_mode: str = "direct", **kwargs: Any
) -> tuple[Any, FunctionDependencies, TraceScopeState]:
    scope = tracer.begin_scope()
    scope.external_process_mode = external_process_mode
    random_state_before = _get_numpy_random_state_b64()

    try:
        result = func(*args, **kwargs)
    except BaseException:
        tracer.end_scope(scope)
        raise

    random_state_after = _get_numpy_random_state_b64()
    random_states: dict | None = None
    if random_state_before is not None and random_state_after != random_state_before:
        random_states = {
            "before": {"numpy": random_state_before},
            "after": {"numpy": random_state_after},
        }

    tracer.end_scope(scope)

    if scope.global_changed:
        for msg in scope.global_change_messages:
            print_change(msg)

    code_dependencies_list = []
    all_qual_names: set[str] = set()
    for func_key, val in scope.code_dependencies_counters.items():
        if val <= 0:
            continue

        qual_parts = scope.function_qualified_name[func_key].split(".")

        # Skip nested functions if parent already listed
        if len(qual_parts) > 1:
            if any(
                func_key.filename + "." + ".".join(qual_parts[:-k]) in all_qual_names for k in range(len(qual_parts))
            ):
                continue

        module = scope.function_modules[func_key]
        qual_name = ".".join(qual_parts)
        all_qual_names.add(func_key.filename + "." + qual_name)

        code_dependencies_list.append(
            CodeDependency(
                function_qualified_name=qual_name,
                module=module,
                filename=func_key.filename,
                bytecode_hash=scope.function_bytecode_hash[func_key],
                global_vars=scope.globals[func_key],
                closure_vars=scope.closure_vars[func_key],
                package_versions=scope.package_versions[func_key],
                compiled_dependencies=scope.compiled_dependencies[func_key],
                file_last_modified_date_str=_get_file_last_modified_date_str(func_key.filename),
            )
        )

    root_dependency = _get_callable_dependency(_require_code_backed_callable(func))
    if not any(
        dep.bytecode_hash == root_dependency.bytecode_hash
        and dep.filename == root_dependency.filename
        and dep.function_qualified_name == root_dependency.function_qualified_name
        for dep in code_dependencies_list
    ):
        code_dependencies_list.insert(0, root_dependency)

    data_dependencies_list = [key for key, val in scope.data_dependencies_counters.items() if val > 0]
    inherited_dependencies_list = [key for key, val in scope.inherited_dependencies_counters.items() if val > 0]

    dependencies = FunctionDependencies(
        code=code_dependencies_list,
        data=data_dependencies_list,
        inherited=inherited_dependencies_list,
        random_states=random_states,
        external_processes=list(scope.external_processes.values()),
        ambient_time_sources=sorted(scope.ambient_time_sources),
        ambient_environment_sources=sorted(scope.ambient_environment_sources),
    )
    return result, dependencies, scope


MemoizeMode = Literal["strict", "safe", "optimistic"]
ExternalProcessMode = Literal["manual", "direct", "strict"]


@overload
def memoize[**P, R](func: Callable[P, R]) -> Callable[P, R]: ...
@overload
def memoize[**P, R](
    *,
    mode: MemoizeMode = ...,
    external_process_mode: ExternalProcessMode = ...,
    ignore_code_changes: bool = ...,
    condition: Callable[..., bool] | None = ...,
    serializer: ResultSerializer[R],
    argument_hasher: ArgumentHasher | None = ...,
    store_call_arguments: bool = ...,
) -> Callable[[Callable[P, R]], Callable[P, R]]: ...
@overload
def memoize[**P, R](
    *,
    mode: MemoizeMode = ...,
    external_process_mode: ExternalProcessMode = ...,
    ignore_code_changes: bool = ...,
    condition: Callable[..., bool] | None = ...,
    argument_hasher: ArgumentHasher | None = ...,
    store_call_arguments: bool = ...,
) -> Callable[[Callable[P, R]], Callable[P, R]]: ...


def memoize[**P, R](
    func: Callable[P, R] | None = None,
    *,
    mode: MemoizeMode = "safe",
    external_process_mode: ExternalProcessMode = "direct",
    ignore_code_changes: bool = False,
    condition: Callable[..., bool] | None = None,
    serializer: ResultSerializer[R] | None = None,
    argument_hasher: ArgumentHasher | None = None,
    store_call_arguments: bool = False,
) -> Callable[P, R] | Callable[[Callable[P, R]], Callable[P, R]]:
    """Memoize a function's results to disk with dependency tracking.

    Supports three modes:
    - "strict": raises RuntimeError when dependency tracking is incomplete
      (e.g., globals()/locals() access detected).
    - "safe" (default): forces recompute when in doubt — never returns stale
      results, uses canonical hashing for deterministic cache keys.
    - "optimistic": caches aggressively, ignores risks — may return stale
      results when globals() is used or pickle is non-deterministic.

        Additional options:
        - external_process_mode: controls how subprocess/os.system dependencies are handled.
            "manual" disables automatic executable tracking, "direct" tracks the launched
            executable path but records that coverage is incomplete, and "strict" raises if
            traced code launches external processes without complete dependency coverage.
        - ignore_code_changes: skip bytecode-based invalidation for code dependencies
            while still checking data dependencies, globals, closures, and random state.
        - condition: when provided, caching is enabled only if condition(*args, **kwargs)
            returns True for the current invocation.
        - serializer: custom result serializer used for cache save/load. Argument hashing
            remains unchanged.
        - argument_hasher: custom cache-key hook for args/kwargs. Useful when arguments
            are large or not picklable.
        - store_call_arguments: when True, save the original args/kwargs to a companion
            pickle file next to the cached result for later replay.
    """

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        active_serializer = serializer or PICKLE_RESULT_SERIALIZER
        code_backed_func = _require_code_backed_callable(func)

        @functools.wraps(func)
        def _memoize_wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            global disk_cache_dir

            if condition is not None and not condition(*args, **kwargs):
                return func(*args, **kwargs)

            co = code_backed_func.__code__

            full_qualified_name = f"{code_backed_func.__module__}.{code_backed_func.__qualname__}"
            full_qualified_name_file = full_qualified_name.replace("<locals>", "locals")

            # Build cache key from function identity, arguments, and closure state.
            # Use a fallback for non-picklable closure variables (e.g., lambdas, locks)
            # so the function still works — just with a less precise cache key.
            closure_nonlocals = inspect.getclosurevars(func).nonlocals
            try:
                closure_key = pickle.dumps(closure_nonlocals)
            except (pickle.PicklingError, TypeError, AttributeError):
                # Fallback: hash the repr + id of each non-picklable value
                closure_key = pickle.dumps({k: f"{type(v).__qualname__}:{id(v)}" for k, v in closure_nonlocals.items()})

            if argument_hasher is not None:
                hash_bytes = (
                    pickle.dumps((co.co_name, co.co_filename))
                    + closure_key
                    + _normalize_argument_hash_value(argument_hasher.hash_args(args, kwargs))
                )
            elif mode == "optimistic":
                # Raw pickle — fast but may produce different hashes for identical objects
                hash_bytes = pickle.dumps((co.co_name, co.co_filename, args, kwargs)) + closure_key
            else:
                # Canonical hashing — deterministic across insertion orders
                hash_bytes = _canonical_hash((co.co_name, co.co_filename, args, kwargs)) + closure_key

            hash_str = hashlib.sha256(hash_bytes).hexdigest()
            hash_str_short = hash_str[:16]
            cache_prefix = os.path.join(disk_cache_dir, f"{full_qualified_name_file}_{hash_str_short}")

            assert len(cache_prefix) + 40 < 260, "long file names not handled yet"
            cache_lock = _get_cache_key_lock(cache_prefix)

            def return_cached(cached_result: R, cached_dependencies: dict[str, Any], dep_file: str) -> R:
                global __last_cache_loading__
                __last_cache_loading__ = full_qualified_name_file
                last_modified_date = os.stat(dep_file).st_mtime
                last_modified_date_str = str(datetime.datetime.fromtimestamp(last_modified_date))
                dep = DataDependency(
                    file_path=dep_file,
                    last_modified_date_str=last_modified_date_str,
                )
                tracer.add_inherited_dependency(dep)

                if numpy is not None:
                    if cached_dependencies["random_states"] is not None:
                        numpy.random.set_state(
                            pickle.loads(
                                binascii.a2b_base64(
                                    cached_dependencies["random_states"]["after"]["numpy"].encode("utf-8")
                                )
                            )
                        )
                return cached_result

            with cache_lock:
                with _cache_process_lock(cache_prefix):
                    cached_result, cached_dependencies, matched_dep_file = _load_matching_cache_entry(
                        func, cache_prefix, active_serializer, mode, external_process_mode
                    )
                    if cached_result is not _CACHE_MISS:
                        assert cached_dependencies is not None
                        assert matched_dep_file is not None
                        result_file = os.path.split(matched_dep_file.replace("_dependencies.json", "_result.pkl"))[1]
                        print(f"Result loaded from {result_file}")
                        return return_cached(cached_result, cached_dependencies, matched_dep_file)

                    result, dependencies, scope = get_dependencies_runtime(
                        func,
                        *args,
                        external_process_mode=external_process_mode,
                        **kwargs,
                    )

                    if dependencies.external_processes:
                        incomplete_external_processes = [
                            dep for dep in dependencies.external_processes if not dep.tracking_complete
                        ]
                        if incomplete_external_processes:
                            process_paths = ", ".join(dep.executable_path for dep in incomplete_external_processes)
                            if external_process_mode == "strict" or mode == "strict":
                                raise RuntimeError(
                                    "memodisk strict mode: external process dependency tracking is incomplete for "
                                    f"{process_paths}. Use add_data_dependency(...) or opt into optimistic/direct "
                                    "behavior explicitly."
                                )
                            if mode == "safe":
                                return result

                    if dependencies.ambient_time_sources:
                        source_names = ", ".join(dependencies.ambient_time_sources)
                        if mode == "strict":
                            raise RuntimeError(
                                "memodisk strict mode: ambient time dependency tracking is incomplete for "
                                f"{source_names}. Pass time-dependent values as explicit arguments or opt into "
                                "optimistic behavior explicitly."
                            )
                        if mode == "safe":
                            return result

                    if dependencies.ambient_environment_sources:
                        source_names = ", ".join(dependencies.ambient_environment_sources)
                        if mode == "strict":
                            raise RuntimeError(
                                "memodisk strict mode: ambient environment dependency tracking is incomplete for "
                                f"{source_names}. Pass environment-dependent values as explicit arguments or opt into "
                                "optimistic behavior explicitly."
                            )
                        if mode == "safe":
                            return result

                    if scope.has_dynamic_globals:
                        funcs = ", ".join(scope.dynamic_globals_functions)
                        if mode == "strict":
                            raise RuntimeError(
                                f"memodisk strict mode: dependency tracking is incomplete because "
                                f"these functions access globals via globals()/locals(): {funcs}"
                            )
                        if mode == "safe":
                            return result

                    if scope.global_changed:
                        print_change("Skipping cache: global variables changed during execution")
                        return result

                    os.makedirs(os.path.dirname(cache_prefix), exist_ok=True)

                    assert len(dependencies.code) > 0
                    all_dependencies: dict[str, Any] = {
                        "arguments_hash": hash_str,
                        "code": [asdict(d) for d in dependencies.code],
                        "data": [asdict(d) for d in dependencies.data],
                        "random_states": dependencies.random_states,
                        "external_processes": [asdict(d) for d in dependencies.external_processes],
                        "ambient_time_sources": dependencies.ambient_time_sources,
                        "ambient_environment_sources": dependencies.ambient_environment_sources,
                        "ignore_code_changes": ignore_code_changes,
                        "result_serializer": active_serializer.name,
                        "argument_hasher": None if argument_hasher is None else argument_hasher.name,
                    }
                    for inherited_dependencies in dependencies.inherited:
                        with builtins_open(inherited_dependencies.file_path, "r") as fh:
                            dependencies_to_add = json.load(fh)
                        for data_dep in dependencies_to_add["data"]:
                            all_dependencies["data"].append(data_dep)
                        for data_dep in dependencies_to_add["code"]:
                            all_dependencies["code"].append(data_dep)
                        for external_process_dep in dependencies_to_add.get("external_processes", []):
                            all_dependencies["external_processes"].append(external_process_dep)
                        for ambient_time_source in dependencies_to_add.get("ambient_time_sources", []):
                            all_dependencies["ambient_time_sources"].append(ambient_time_source)
                        for ambient_environment_source in dependencies_to_add.get("ambient_environment_sources", []):
                            all_dependencies["ambient_environment_sources"].append(ambient_environment_source)

                    all_dependencies["ambient_time_sources"] = sorted(set(all_dependencies["ambient_time_sources"]))
                    all_dependencies["ambient_environment_sources"] = sorted(
                        set(all_dependencies["ambient_environment_sources"])
                    )

                    dep_hash = hashlib.sha256(json.dumps(all_dependencies, sort_keys=True).encode()).hexdigest()[:12]

                    result_file = f"{cache_prefix}_{dep_hash}_result.pkl"
                    dependencies_file = f"{cache_prefix}_{dep_hash}_dependencies.json"

                    call_arguments_file: str | None = None
                    if store_call_arguments:
                        call_arguments_file = f"{cache_prefix}_{dep_hash}_call_arguments.pkl"
                        _serialize_big_data_atomic(
                            {"args": args, "kwargs": kwargs},
                            call_arguments_file,
                            PICKLE_RESULT_SERIALIZER,
                        )

                    all_dependencies["store_call_arguments"] = store_call_arguments
                    all_dependencies["call_arguments_file"] = call_arguments_file

                    _serialize_big_data_atomic(result, result_file, active_serializer)
                    _write_json_atomic(dependencies_file, all_dependencies)

            return result  # ty

        return _memoize_wrapper

    if func is not None:
        # @memoize (bare decorator, no parentheses)
        return decorator(func)
    # @memoize() or @memoize(mode="strict")
    return decorator


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

    try:
        yield None
    finally:
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


def _is_read_only_open_mode(mode: str) -> bool:
    return "+" not in mode and all(flag not in mode for flag in ("w", "a", "x"))


def _extract_open_target_and_mode(
    file: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> tuple[str | None, str]:
    mode = kwargs.get("mode")
    if mode is None and args:
        mode = args[0]
    if not isinstance(mode, str):
        mode = "r"

    path: str | None = None
    if isinstance(file, str | bytes | os.PathLike):
        path = str(file)

    return path, mode


def memoized_open_wrapper(file: Any, *args: Any, **kwargs: Any) -> IO[Any]:
    filename, mode = _extract_open_target_and_mode(file, args, kwargs)
    if filename is not None and _is_read_only_open_mode(mode):
        if not os.path.exists(filename):
            raise FileNotFoundError(filename)

        with loop_until_access_time_greater_than_modification_time(filename):
            return builtins_open(file, *args, **kwargs)

    return builtins_open(file, *args, **kwargs)


def memoized_path_open_wrapper(self: pathlib.Path, *args: Any, **kwargs: Any) -> IO[Any]:
    filename, mode = _extract_open_target_and_mode(self, args, kwargs)
    if filename is not None and _is_read_only_open_mode(mode):
        if not os.path.exists(filename):
            raise FileNotFoundError(filename)

        with loop_until_access_time_greater_than_modification_time(filename):
            return pathlib_Path_open(self, *args, **kwargs)

    return pathlib_Path_open(self, *args, **kwargs)


class DataLoaderWrapper[RetType]:
    """Wrap the function for the input file to be added as data dependency."""

    def __init__(self, fun: Callable[..., RetType], position: int = 0):
        self.fun = fun
        self.position = position

    def __call__(self, *args: Any, **kwargs: Any) -> RetType:
        filename = args[self.position]
        with loop_until_access_time_greater_than_modification_time(filename):
            out = self.fun(*args, **kwargs)
        return out

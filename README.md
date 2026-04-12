# memodisk

A Python package to cache function results on disk with dependency changes tracking.

![Python package](https://github.com/martinResearch/memodisk/workflows/Python%20package/badge.svg)

## Goal

This package provides a Python decorator to save on disk and reuse the results of functions that are long to execute. This can be referred to as *persistent memoization*.

The result of a decorated function is loaded from disk if and only if

* the function has been called previously with the same input argument
* there are no changes in the Python code dependencies or data dependency files.

The use of the second condition differentiates this library from most other Python persistent memoization libraries. It is a useful feature when prototyping code with frequent code changes.

## Warning

This is a prototype with limited testing and currently targets Python 3.12+. There could be security risks related to the use of Pickle. Some data, code or global variables dependencies could be not detected leading the memoization to return stale results (see the limitations section). If you find failure modes that are not listed in the limitations, please create an issue with a minimal code example to reproduce the problem.

## Installation

```bash
pip install memodisk
```

## Examples

Using the memoization is as simple as adding the `@memoize` decorator to the function you want to use memoization with. In general you want to use memoization on a function whose execution time is longer than the time it takes to check that this function has already been called with the same argument (compute the input hashes) and load the result from disk.
To get the largest speedups thanks to memoization, it might be necessary to refactor the code to move the parts that are long to execute into functions that take limited size inputs.

The decorator also accepts options such as `mode=...`, `external_process_mode=...`, `ignore_code_changes=True`, `condition=...`, `serializer=...`, `argument_hasher=...`, and `store_call_arguments=True`. The `condition` callable receives the same arguments as the memoized function and allows caching only for selected invocations. The `serializer` option customizes how cached results are written and read back, while argument hashing remains unchanged. The `argument_hasher` option customizes cache-key generation for `args` and `kwargs`, which is useful for very large inputs or objects that are not picklable by default. The `store_call_arguments` option writes a companion pickle file containing the original `args` and `kwargs` so a cached call can be replayed later. The helper `load_cached_call_arguments(...)` reads those stored arguments back from a dependencies JSON file. The `external_process_mode` option controls subprocess and `os.system` handling: `manual` disables automatic executable tracking, `direct` records the launched executable path but marks subprocess coverage as incomplete, and `strict` raises when traced code launches external processes without complete dependency coverage. The `mode` policy also governs common ambient time APIs such as `time.time()` and `datetime.datetime.now()`: `safe` recomputes and skips cache writes, `strict` raises, and `optimistic` allows caching.

### Example 1: code dependency changes

let's run the code in [example_1.py](./tests/example_1.py) several times.

```python
from memodisk import memoize


def fun_a(x):
    print("executing fun_a")
    return x * x * 3


@memoize
def fun_b(x):
    print("executing fun_b")
    return fun_a(x) + 2


if __name__ == "__main__":
    print(f"fun_b(5) = {fun_b(5)}")
```

The first time we get

```python
executing fun_b
executing fun_a
fun_b(5) = 77
```

The second time we get

```
Result loaded from fun_b from
C:\Users\martin\AppData\Local\Temp/memodisk_cache/fun_b_0a2333051d1ac2dd_result.pkl
fun_b(5) = 77
```

As you can see the function fun_b is executed only once. The second time the function is called, the result is loaded from a file on disk and the function is not executed.

Let's now edit the file and replace `x * x * 3` by `x * x * 2` and execute again. As expected we now get

```
executing fun_b
executing fun_a
fun_b(5) = 52
```

The change in the body of the function `fun_a`, that is a dependency of `fun_b`, has been detected automatically.

### Example 2: data dependency changes

The example [example_2.py](./tests/example_2.py) illustrates how to keep track of data dependencies access using the built-in `open` function.

```python
from memodisk import memoize


def save_file(x):
    print("save_file")
    fh = open("data_file.txt", "w")
    fh.write(str(x))


@memoize
def load_file():
    print("load_file")
    fh = open("data_file.txt", "r")
    line = fh.readline()
    return line


if __name__ == "__main__":
    save_file("a")
    assert load_file() == "a"
    save_file("b")
    assert load_file() == "b"
```

When we call `save_file("b")` we overwrite the data in `data_file.txt`. This change in the file content gets detected when calling `load_file` for the second time. This is done by automatically replacing the built-in Python function *open* with a wrapper around this function that keeps track of files that are accessed for reading.

### Example 3: file access monkey patching

The built-in `open` function is not the only way the code can access data. For example images can be loaded using opencv's `imread` function. If some data is loaded with another function than the built-in `open` function then the data dependency will not be automatically detected.

We provide a function `add_data_dependency` that the user can call from their code next to the line of code that loads the data, with the path of the file that contains the data as input. However this can be error prone as it is very easy to forget calling `add_data_dependency` in some places.

We provide a less error-prone mechanism, through a functor called `DataLoaderWrapper`. The functor allows the user to replace any function accessing data (opencv's `imread` function for example) by a wrapper around this function so that it automatically calls the function add_data_dependency each time `imread` is used. This in-memory modification is called *monkey-patching* and is done in [example_3.py](./tests/example_3.py) using the `DataLoaderWrapper` functor.

```python
from memodisk import memoize, DataLoaderWrapper
import cv2

# wrap the function for the input file to be added as data dependency
cv2.imread = DataLoaderWrapper(cv2.imread)
```

## How this works

For each function that is decorated with the `memoize` decorator we keep track of all the Python functions it depends on at runtime. Similarly to what is done in the [coverage](https://coverage.readthedocs.io/en/6.0.2/), we register a callback with Python's runtime monitoring API (`sys.monitoring` in Python 3.12+) so the interpreter notifies memodisk when Python functions start executing. This runtime approach captures only the functions actually executed for the given arguments, avoiding the false positives inherent in static analysis (see [Runtime vs Static Dependency Analysis](#runtime-vs-static-dependency-analysis) for a detailed comparison). In order to keep the list of dependency files reasonable we exclude from this list of dependencies the functions defined in files under the Python lib folder, assuming these will not get modified. The user can also provide an additional list of files to exclude.

For each function listed in the dependencies we compute a hash from its bytecode.
Using the bytecode instead of the function body text allows the user to modify the comments or the formatting of the function without invalidating the previously cached results for this function or the functions that depend on it. We could use instead the hash of the Abstract Syntax Tree of the function (see the [ast module](https://docs.python.org/3/library/ast.html)), but that would rely on the assumption that the code source is not modified during execution of the script (unless there is a way to get access to the AST from when the execution was started). On current Python 3.12+ with `debugpy`, line breakpoints did not reproduce a bytecode hash change in our tests.
We also store the source file modification time for each code dependency so cache validation can skip recomputing bytecode hashes when the dependency file has not changed.
For globals or imported callables backed by packages installed in `site-packages`, we also store the distribution version so upgrading a dependency invalidates stale cache entries even when the package code itself is not traced as a user dependency.
For compiled extension modules that are already loaded under those same top-level packages, we also store the extension file modification times so in-place `.pyd`/`.so`/`.dll` swaps invalidate the cache.

We keep track of the data dependencies by storing the last modification date of the files that have been opened in read mode.
When traced code launches external processes through `subprocess` or `os.system`, memodisk can also store the resolved top-level executable path as a data dependency so changing that executable invalidates stale cache entries. This is only direct executable tracking: memodisk does not attempt to discover the full runtime dependency graph of that process such as transitively loaded `.dll`/`.so` files, plugins, shell expansion, or other environment-dependent behavior. Cache entries also record whether external-process tracking was complete. In `safe` mode, incomplete external-process coverage forces recomputation instead of serving a cached result, and in `strict` mode it raises.
When traced code calls common ambient time APIs such as `time.time()`, `time.time_ns()`, `datetime.datetime.now()`, `datetime.datetime.utcnow()`, `datetime.datetime.today()`, or `datetime.date.today()`, memodisk records that the result depended on ambient time. This tracking is intentionally treated as incomplete: in `safe` mode memodisk recomputes instead of serving or writing a cache entry, and in `strict` mode it raises. The robust way to memoize time-sensitive code is to pass the relevant timestamp or date in as an explicit function argument.
The code dependencies hashes and data dependencies last modification dates are saved in a human readable json file in the folder `memodisk_cache` in the user's temp folder (this can be modified at the module level by changing the module variable `disk_cache_dir` ) while the result of the function is saved in a binary file using pickle by default or a custom serializer when provided.
The names of the two generated files differ only by the extension (json and pkl) and are formatted
by default as `{function_name}_{arguments_hash}.json` and `{function_name}_{arguments_hash}.pkl`.
There are some subtle details to take into consideration when accessing data from a file in order to guarantee that the caching will not provide stale results.
The data dependency "version" is obtained by recording the modification date of the accessed file. The modification date resolution is limited and it is possible that a file gets modified during the lapse of time during which the modification date is not incremented. We guard against this by locking the file for a time that is greater than the modification date quantization step.

The hash of the arguments is obtained by pickling the arguments. This can be slow if the input is large or made of many small objects.

Here is an example of a generated dependencies json file:

```json
{
    "arguments_hash": "bc039221ed77e5262e42baaa0833d5fe43217faae894c75dcde3025bf4a1282e",
    "code": [
        {
            "function_qualified_name": "load_file_using_context_manager",
            "module": "__main__",
            "filename": "D:\\repos\\memodisk\\tests\\test_data_dependency_change.py",
            "bytecode_hash": "f7b971c6ea7997dc5c5222f74cec2a249e8680293511c6a8350b621643af2d07",
            "global_vars": {},
            "closure_vars": {},
            "package_versions": {},
            "compiled_dependencies": {},
            "file_last_modified_date_str": "2026-04-04 15:33:14.682488"
        }
    ],
    "data": [
        {
            "file_path": "C:\\test_file.txt",
            "last_modified_date_str": "2020-03-04 15:33:14.682488"
        }
    ],
    "random_states": null,
    "external_processes": [],
    "ambient_time_sources": [],
    "ambient_environment_sources": [],
    "ignore_code_changes": false,
    "result_serializer": "pickle",
    "argument_hasher": null,
    "store_call_arguments": false,
    "call_arguments_file": null
}
```

We do not try to detect changes in global variables but an error will be raised if any of the functions listed in the dependencies uses global variables.

## Runtime vs Static Dependency Analysis

A key architectural choice for any persistent memoization library is how code dependencies are discovered. There are two main approaches:

**Runtime monitoring** (used by memodisk) registers callbacks with the interpreter (`sys.monitoring` in Python 3.12+) to observe which functions are actually invoked during execution. This captures the exact set of dependencies for each call with its specific arguments.

- **Pros:** No false positives from untaken code paths — if a branch is not executed for the given arguments, its dependencies are not recorded. Detects dependencies reached through dynamic dispatch, `getattr`, registry lookups, and conditional imports.
- **Cons:** Requires executing the function at least once to establish the dependency set. The first call incurs tracing overhead. If a code path is taken in some calls but not others, the dependency set varies per call.

**Static inspection** (used by checkpointer, charmonium.cache) analyzes the function's source, bytecode, global scope, type annotations, or object constructions at decoration time or import time to determine dependencies without running the function.

- **Pros:** The dependency set is available immediately without executing the function. It is deterministic and does not vary across calls.
- **Cons:** May include dependencies from code paths not actually taken for a given set of arguments, leading to unnecessary cache misses when one of these unused dependencies changes. May miss dependencies reached through dynamic dispatch, `getattr`, registry lookups, or conditional imports.

Both approaches track code changes via hashing (bytecode hash for memodisk, source or object hash for static tools). The trade-off is essentially **precision vs eagerness**: runtime monitoring is more precise per-call but deferred, while static inspection is eager but may over- or under-approximate.

## Using numpy's random generator

A function that uses the default numpy generator is problematic for caching for two reasons: 1) its output depends on the random generator state that is not provided as an explicit input to the function 2) the function modifies the state of the random generator for the functions that get called after it.
When retrieving cached results for such a function we want to use the state of the random generator when entering the function in the hash of the inputs and after retrieving cached results we want to set the random state to the same state as the one we would get by running the cached function. 
The use of the random generator is detected by comparing the state of the random generator state before and after executing the function.
The input state and output state of the random generator are saved in the json file and the memoized result is loaded in subsequent run only if the random state is identical to the one saved in the json file i.e. when entering the function at the first run, the result is loaded from the pickle file and the random state is modified to match the random state after execution of the function at the first run.

This mechanism can fail in some cases, if a function accesses the random generator but restores the generator in the same state as it was when entering the function for example and thus we recommend to avoid using the default "global" numpy random generator, but instead to use instances of `numpy.random.Generator` that are passed as arguments to the functions that use the random number in order to reduce the risk of getting stale results from the memoize decorator.

If the same function is called multiple times with the same input arguments but with different random states, then a single memoization file is used and gets overwritten. We could add an argument to the memoize decorator to tell the memoize decorator to use the random state when computing the hash of the input arguments to allow the use of multiple memoization files for the same function with one file for each state of the random generator.

## Limitations

* by default, requires all the function arguments of the memoized function to be serializable using pickle. This can be overridden with `argument_hasher=...` for cache-key generation.
* may not detect all global variables dependencies.
* ambient time detection covers common direct calls such as `time.time()` and `datetime.datetime.now()`, but may still miss indirect wrappers, custom clocks, or time values obtained outside the memoized call and captured through globals or closures.
* external-process tracking is limited to the directly resolved executable path. It does not cover full transitive native dependencies such as `.dll`/`.so` chains, plugins, registry/config driven behavior, or other runtime environment changes.
* may miss some shell-invoked external commands when the real executable cannot be resolved from the command string. In those cases the executable should still be declared explicitly with `add_data_dependency`.
* does not detect changes in remote dependencies fetched from the network.
* has no configurable cache size.
* will memoize only the decorated functions. 

Some of these failure modes can be reproduced using scripts in the [failure_modes](./failure_modes) folder.

## Potential Python improvements

memodisk is intentionally built on standard CPython mechanisms rather than a custom interpreter, so a few targeted Python/runtime improvements could make it both faster and more robust:

* a more selective extension of `sys.monitoring` (PEP 669) that can trace only the dynamic extent of a memoized call and report higher-level dependency events with lower overhead than per-frame or per-opcode callbacks
* a first-class interpreter API for observing important side effects and ambient inputs such as file opens, subprocess launches, environment-variable reads, time APIs, and randomness sources
* stable interpreter-provided fingerprints for code objects, imported modules, and installed-package versions so tools like memodisk do less ad hoc hashing and dependency bookkeeping themselves

These would not change memodisk's core design, but they would reduce tracing overhead, improve coverage of non-Python dependencies, and simplify parts of the implementation.

## TODO

* improve the detection of non-pure function so that it works when using a compiled third party module
* experimental strict declared-dependency mode — allow a memoized function to declare the files or directory roots it is allowed to read, then execute the outermost memoized call in an isolated subprocess sandbox that fails on undeclared file access. Nested memoized calls should inherit the active sandbox by default and only be allowed to narrow the allowlist, not expand it.
* add less intrusive alternatives to the decorator-only API — e.g. registering functions in a list of function names provided directly to `disk_memoize`, or other ways to opt into memoization without editing every function definition
* implement an automatic memoization of function that are long to evaluate using similar criterion to IncPy (see references) to decide if a function should be memoize or not
* quantify runtime overhead with benchmarks — separate empty-cache tracing cost, cache-hit validation/load cost, and serialization cost; determine the break-even point as a function of execution time and result size
* async function support (offered by checkpointer, perscache)
* TTL-based cache expiry — automatically invalidate entries after a time-to-live period (offered by checkpointer, perscache)
* selective argument ignoring — exclude specific arguments from cache key computation (offered by checkpointer via `HashBy`/`NoHash`, perscache via `ignore`)
* configurable cache size limits and automatic cache size management — evict entries when total storage exceeds a user-defined threshold, starting with LRU; consider smarter policies like GDSF (Greedy-Dual-Size-Frequency) that factor in recompute cost and storage size, not just recency (offered by charmonium.cache)
* overhead-aware caching — measure time saved vs memoization overhead per function and warn when caching is not worthwhile (offered by charmonium.cache via `verbose=True` exit report)
* pluggable serialization formats — JSON, YAML, CSV, Parquet alongside pickle (offered by perscache)
* remote/cloud storage backends — S3, GCS, SFTP (offered by perscache, provenance)
* lineage tracking and provenance visualization — record which function produced which artifact from which inputs, with queryable metadata (offered by provenance)
* class method caching with per-instance cache isolation (offered by perscache)

## Alternatives

* [checkpointer](https://github.com/Reddan/checkpointer). Decorator-based persistent memoization with automatic code-aware cache invalidation. Uses **static inspection** to discover code dependencies (see [Runtime vs Static Dependency Analysis](#runtime-vs-static-dependency-analysis) for the trade-offs vs memodisk's runtime approach). checkpointer supports async functions, in-memory and custom storage backends, TTL-based expiry, captured global variables (`CaptureMe`/`CaptureMeOnce`), and fine-grained argument hashing via `HashBy`/`NoHash` type annotations. memodisk additionally tracks data file dependencies (files opened for reading), external process invocations, ambient time API usage, numpy random generator state, and installed package versions. checkpointer does not detect data dependency changes.
* [perscache](https://github.com/leshchenko1979/perscache). Decorator-based persistent memoization with pluggable serialization (JSON, YAML, pickle, Parquet, CSV) and storage backends (local disk, Google Cloud Storage). Invalidation is based on hashing the decorated function's own code and arguments, but it does **not** trace transitive dependencies — if function `A` calls function `B` and `B` changes, the cache for `A` is not invalidated. memodisk's runtime monitoring captures the full call graph and detects such changes automatically. perscache additionally offers async support, instance-specific class method caching, TTL-based expiry, selective argument ignoring, and automatic LRU cleanup when storage exceeds a size threshold. It does not track data file dependencies or external process invocations.
* [provenance](https://github.com/bmabey/provenance). Caching and **lineage tracking** library designed for ML pipelines. Results are wrapped in `ArtifactProxy` objects that record full provenance graphs (which function produced which artifact from which inputs), stored in a Postgres artifact repository with blobs on disk, S3, or SFTP. This lineage visualization and queryable metadata store distinguishes it from pure memoization libraries. However, provenance does **not** detect code changes automatically — when a function's implementation changes, the user must manually bump the `version` parameter in the decorator to force recomputation. It also does not track data file dependencies. memodisk detects both code and data dependency changes at runtime without manual intervention. Note: the repository was archived in February 2026.
* [mandala](https://github.com/amakelov/mandala). Memoization + **computation graph** framework (`@op` decorator) that persists call inputs/outputs in a SQLite-backed storage and organizes them into queryable `ComputationFrame` graphs — a relational structure where each `@op` has a table of calls and variables are linked by foreign keys. Code changes are detected via content hashing of function source and dependencies, but **version compatibility is managed by the user**: when a dependency changes, you decide whether the change is backward-compatible or invalidating. This is more flexible than fully automatic invalidation but requires manual judgment. mandala does not track data file dependencies or external process invocations. Compared to memodisk, mandala's strengths are its structured computation graph with dataframe extraction, collection-level tracking (individual list/dict elements), and composable end-to-end pipelines. memodisk's strengths are fully automatic invalidation (no user intervention on code changes) and tracking of data files, subprocesses, time APIs, random state, and package versions. Published as `pymandala` on PyPI; alpha status.
* [Cachier](https://github.com/shaypal5/cachier). Does not seem to have any mechanism to detect change in code or data dependencies. The only mechanism to invalidate stale cached data is through providing expiration dates.
* [IncPy](https://github.com/pajju/IncPy). Among the surveyed alternatives, IncPy is the closest to memodisk's approach: it uses **runtime tracing** through a modified CPython 2.6 interpreter to automatically detect which functions and files a computation depends on — no decorators, no static analysis. Because dependency discovery happens at execution time, it shares memodisk's key advantage of no false positives from untaken branches or unused imports (see [Runtime vs Static Dependency Analysis](#runtime-vs-static-dependency-analysis)). The main differences are: (1) IncPy requires a custom interpreter fork, while memodisk uses the standard `sys.monitoring` API (PEP 669) available in any CPython 3.12+; (2) IncPy memoizes transparently (every function call is a candidate), while memodisk requires an explicit `@memoize` decorator; (3) IncPy is limited to Python 2.6 and has been unmaintained for over a decade. Papers: *Towards practical incremental recomputation for scientists: An implementation for the Python language* ([pdf](https://www.usenix.org/legacy/events/tapp10/tech/full_papers/guo.pdf)), *Using Automatic Persistent Memoization to Facilitate Data Analysis Scripting* — Philip J. Guo and Dawson Engler, Stanford University.
* [charmonium.cache](https://pypi.org/project/charmonium.cache). Uses **static analysis** to discover code dependencies (see [Runtime vs Static Dependency Analysis](#runtime-vs-static-dependency-analysis) for the trade-offs). Also offers bounded cache size with GDSF (Greedy-Dual-Size-Frequency) eviction, overhead-aware caching reports, `FileContents` for data dependency tracking (requires code modifications), a CLI for memoizing shell commands, and distributed/network cache support via `PathLike` custom stores. The file names used in the cache folder are hash-based and not easily interpretable.
* [cache-to-disk](https://pypi.org/project/cache-to-disk/). Python decorator to memoize function to disk. It does not detect changes in the code or data dependencies. Cached data can be given an expiry date and it is possible to invalidate all cached results for a function using a simple command.
* [joblib.Memory](https://joblib.readthedocs.io/en/latest/generated/joblib.Memory.html). Python decorator to memoize function to disk. It does not detect changes in the code or data dependencies.
* [Artemis.fileman.disk_memoize](https://github.com/QUVA-Lab/artemis/blob/master/artemis/fileman/disk_memoize.py) It does not detect changes in the code or data dependencies. [pdf](http://citeseerx.ist.psu.edu/viewdoc/download;jsessionid=59BEC4646686E70CFD2428EF9786B9D0?doi=10.1.1.224.164&rep=rep1&type=pdf)
* [noWorkflow](http://gems-uff.github.io/noworkflow/). *noWorkflow: a Tool for Collecting, Analyzing, and Managing Provenance from Python Scripts* [pdf](https://par.nsf.gov/servlets/purl/10048452). Library that allows to track how data has been generated. It bears some similarity with the library as it also requires to keep track of dependencies.
* [klepto](https://mmckerns.github.io/project/pathos/wiki/klepto.html). Allows caching of python function results to files or database archive. The detection of code change is not mentioned.
* [exca](https://github.com/facebookresearch/exca). Developed by Facebook Research. The detection of code change is not mentioned.

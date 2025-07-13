# memodisk

**memodisk** is a python module to memoize function results on disk with python code and data dependencies tracking.

![Python package](https://github.com/martinResearch/memodisk/workflows/Python%20package/badge.svg)

## Goal

This package provides a python decorator to save on disk and reuse the results of functions that are long to execute. This can be referred to as *persistent memoization*.

The result of a decorated function is loaded from disk if and only if

* the function has been called previously with the same input argument
* there are no changes in the python code dependencies or data dependency files.

The use of the second condition differentiates this library from most other python persistent memoization libraries. It is a useful feature when prototyping code with frequent code changes.

## Warning

This is a prototype with limited testing and works with python 3.10 only. There could be security risks related to the use of Pickle. Some data, code or global variables dependencies could be not detected leading the memoization to return stale results (see the limitations section). If you find failure modes that are not listed in the limitations, please create an issue with a minimal code example to reproduce the problem.

## Installation

```bash
pip install memodisk
```

## Examples

Using the memoization is as simple as adding the `@memoize` decorator to the function you want to use memoization with. In general you want to use memoization on a function whose execution time is longer than the time it takes to check that this function has already been called with the same argument (compute the input hashes) and load the result from disk.
To get the largest speedups thanks to memoization, it might be necessary to refactor the code to move the parts that are long to execute into functions that take limited size inputs.

### Example 1: code dependency changes

let run the code in [example_1.py](./tests/example_1.py) several times.

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

Let now edit the file and replace `x * x * 3` by `x * x * 2` and execute again. As expected we now get

```
executing fun_b
executing fun_a
fun_b(5) = 52
```

The change in the body of the function `fun_a`, that is a dependency of `fun_b`, has been detected automatically.

### Example 2: data dependency changes

The example [example_2.py](./tests/example_2.py) illustrates how to keep track of data dependencies access using the build-in `open` function.

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

When we call `save_file("b")` we overwrite the data in `data_file.txt`. This change in the file content gets detected when calling `load_file` for the second time. This is done by automatically replacing the built-in python function *open* with a wrapper around this function that keeps track of files that are accessed for reading.

### Example 3: file access monkey patching

The built-in `open` function is not the only way the code can access data. For example images can be loaded using opencv's `imread` function. If some data is loaded with another function than the build-in `open` function then the data dependency will not be automatically detected.

We provide a function `add_data_dependency` that the user can call from his code next to the line of code that loads the data, with the path of the file that contains the data as input. However this can be error prone as it is very easy to forget calling `add_data_dependency` in some places.

We provide a less error-prone mechanism, through a functor called `DataLoaderWrapper`. The functor allows the user to replace any function accessing data (opencv's `imread` function for example) by a wrapper around this function so that it automatically calls the function add_data_dependency each time `imread` is used. This in-memory modification is called *monkey-patching* and is done in [example_3.py](./tests/example_3.py) using the `DataLoaderWrapper` functor.

```python
from memodisk import memoize, DataLoaderWrapper
import cv2

# wrap the function for the input file to be added as data dependency
cv2.imread = DataLoaderWrapper(cv2.imread)
```

## How this works

For each function that is decorated with the `memoize` decorator we keep track of all the python functions it depends on at runtime. Similarly to what is done in the [coverage](https://coverage.readthedocs.io/en/6.0.2/), we provide through `sys.settrace` a callback function to the python interpreter that gets called each time the interpreter calls a new function or executes a new line of code. Doing run time analysis allows us to keep as dependencies only the functions that are required to evaluate the memoize function for the specific arguments. In contrast, static analysis would yield unnecessary dependencies by detecting dependencies for all possible code paths, even the one not executed for the specific set of input arguments. In order to keep the list of dependency files reasonable we exclude from this list of dependencies the functions defined in files under the python lib folder, assuming these will not get modified. The user can also provide an additional list of files he wants to exclude.

For each function listed in the dependencies we compute a hash from its bytecode
Using the bytecode instead of the function body text allows the user to modify the comments or the formatting a the function without invalidating the previously cached results for this function or the functions that depends on it. We could use instead the hash of the Abstract Syntax Tree of the function (see the [ast module](https://docs.python.org/3/library/ast.html)), but that would rely on the assumption that the code source is not modified during execution of the script (unless there is a way to get access to the AST from when the execution was started). One current limitation resulting from hashing the bytecode is that the python debugger modifies the bytecode when adding breakpoint, which leads to cache misses. This could be potentially resolved by filtering out the lines added by the python debugger before computing the hash of a function.

We keep track of the data dependencies by storing the last modification date of the files that have been opened in read mode.
The code dependencies hashes and data dependencies last modification dates are saved in a human readable json file in the folder `memodisk_cache` in the user's temp folder (this can be modified at the module level by changing the model variable `disk_cache_dir` ) while the result of the function is pickled in a binary file.
The names of the two generated files differ only by the extension (json and pkl) and are formatted
by default as `{function_name}_{arguments_hash}.json` and `{function_name}_{arguments_hash}.pkl`.
There are some subtle details to take into consideration when accessing data from a file in order to guarantee that the caching will not provide stale results.
The data dependency "version" is obtained by recording the  modification date of the accessed file. The modification date resolution is limited and it is possible that a file gets modified during the lapse of time during which the modification date is not incremented. We guard against this by locking the file for a time that is greater than the modification date quantization step.

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
            "closure_vars": {}
        }
    ],
    "data": [
        {
            "file_path": "C:\\test_file.txt",
            "last_modified_date_str": "2020-03-04 15:33:14.682488"
        }
    ],
    "random_states": null
}
```

We do not try to detect changes in the global variables but an error will be raised if any of the functions listed in the dependencies uses global variables.

## Using numpy's random generator

A function that uses the default numpy generator is problematic from caching two reasons: 1) its output depends on the random generator state that is not provided as an explicit input to the function 2) the function modifies the state of the random generator for the functions that get called after it.
When retrieving a cached results for such a function we want to use the state of the random generator when entering the function in the hash of the inputs and after retrieveing cached results we want to set the random state to the same state as the one we would get by running the the cached function. 
The use of the random generator is detected by comparing the state of the random generator state before and after executing the function.
The input state and output state of the random generator are saved in the json file and the memoized result is loaded in subsequent run only if the random state is identical to the one saved in the json file i.e. when entering the function at the first run, the result is loaded from the pickle file and the random state is modified to match the random state after execution of the function at the first run.

This mechanism can fail in some cases, if a function access the random generator but restore the generator in the same state as it was when entring the function for example and thus we recommend to avoid using the default "global" numpy random generator, but instead to use instances of `numpy.random.Generator` that are passed as arguments to the functions that use the random number in order to reduce the risk of getting stale results from the memoize decorator.

If the same function is called multiple times with the same input arguments but with different random states, then a single memoization file is used and gets overwritten. We could add an argument to the memoize decorator to tell the memoize decorator to use the random state when computing the hash of the input arguments to allow the use of multiple memoization files for the same function with one file for each state of the random generator.

## Limitations

* does not support the property decorator in some cases as `getattr` is trying to execute the function.
* requires all the function arguments of the memoized function to be serializable using pickle.
* may not detect all global variables dependencies.
* will detect an added breakpoint in a function as a change in the code because the python debugger adds line in the function bytecode when using breakpoint.
* does not detect non determinism due to use of time.
* does not detect changes in C/C++ extension modules or external executables, unless the pyd file, dll or executable dependency is explicitly specified through the `add_data_dependency` function.
* does not detect changes in remote dependencies fetched from the network.
* is not thread safe. It does not support multi-threading
* will not detect if an aliased import is modified.
* computes argument hash using pickled object strings, which does not always produce the same string for identical objects. Could use [compute_fixed_hash](https://github.com/QUVA-Lab/artemis/blob/84d3b1daf0de363cc823d99f978e2861ed400b5b/artemis/general/hashing.py#L25).
* has no configurable cache size.
* will memoize only the decorated functions. 

Some of these failure modes can be reproduced using scripts in the [failure_modes](./failure_modes) folder.

## TODO

* add the module name of the functions in the code dependencies description.
* filter out the lines added by the python debugger when using breakpoint before computing the hash of a function.
* add the file modification date in the code dependencies and a test to skip checking functions hashes if the file modification date did not change.
* add an option to save the full input arguments in the pickle file in order to be able to re-run a function directly from the pickled data
* save the versions of module dependencies that in the lib/site-packages folder and add an option to remove function dependencies from packages under site-packages and use only module version to detect a dependency change, assuming the package in site-packages does not get modified. maybe use the file modification date for code under site-packages instead of functions ast
* improve the detection of non-pure function so that it works when using a compiled third party module
* allow the use of a different serialization library than pickle. It could be provided at module level to disc_memoize or as argument to the memoize decorator
* add the possibility to provide a condition in the decorator to memoize or not
* add a less intrusive alternative to the use of decorator by registering a function in a list of function names provided directly to disc_memoize
* implement an automatic memoization of function that are long to evaluate using similar criterion to IncPy (see references) to decide if a function should be memoize or not
* make the tool thread-safe
* see if we can detect compiled module loading and compiled module calling to add the compiled module as dependency.
* publish module on [pypi.org](pypi.org)
* see if we can make the hashing more deterministic using the method implemented in [charmonium.cache](https://pypi.org/project/charmonium.cache).

## Alternatives

* [Cachier](https://github.com/shaypal5/cachier). Does not seem to have any mechanism to detect change in code or data dependencies. The only mechanism to invalidate stale cached data is through providing expiration dates.
* [IncPy](https://github.com/pajju/IncPy). Enhanced Python interpreter that speeds up script execution times by automatically memoizing (caching) the results. Supports only Python2.6. Paper *Towards practical incremental recomputation for scientists: An implementation for the Python language*. Philip J. Guo and Dawson Engler Stanford University [pdf](https://www.usenix.org/legacy/events/tapp10/tech/full_papers/guo.pdf). *Using Automatic Persistent Memoization to Facilitate Data Analysis Scripting* Philip J. Guo and Dawson Engler
* [charmonium.cache](https://pypi.org/project/charmonium.cache). Code dependencies are obtained by static analysis. This may lead to the inclusion in the list of dependencies of functions that are actually not on the code path execution when calling the memoized function with the given input arguments, and thus may lead to unnecessary cache misses when one of these unused functions is modified. Detecting data dependencies changes requires code modifications using the FileContents class. The file names used in the cache folder are not easily interpretable which makes it harder to manage manually the cached data.
* [cache-to-disk](https://pypi.org/project/cache-to-disk/). Python decorator to memoize function to disk. It does not detect changes in the code or data dependencies. Cached data can be given an expiry date and it is possible to invalidate all cached results for a function using a simple command.
* [joblib.Memory](https://joblib.readthedocs.io/en/latest/generated/joblib.Memory.html). Python decorator to memoize function to disk. It does not detect changes in the code or data dependencies.
* [Artemis.fileman.disk_memoize](https://github.com/QUVA-Lab/artemis/blob/master/artemis/fileman/disk_memoize.py) It does not detect changes in the code or data dependencies. [pdf](http://citeseerx.ist.psu.edu/viewdoc/download;jsessionid=59BEC4646686E70CFD2428EF9786B9D0?doi=10.1.1.224.164&rep=rep1&type=pdf)
* [noWorkflow](http://gems-uff.github.io/noworkflow/). *noWorkflow: a Tool for Collecting, Analyzing, and Managing Provenance from Python Scripts* [pdf](https://par.nsf.gov/servlets/purl/10048452). Library that allows to track how data has been generated. It bears some similarity with the library as it also requires to keep track of dependencies.
* [klepto](https://mmckerns.github.io/project/pathos/wiki/klepto.html). Allows caching of python function results to files or database archive. The detection of code change is not mentioned.

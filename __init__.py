"""Module to memoize function results on disk with python dependencies tracking."""

__version__ = "0.0.1"
__all__ = [
    "memoize",
    "add_data_dependency",
    "DataLoaderWrapper",
    "get_globals_from_code",
    "set_cache_dir",
    "open_delay",
    "get_function_qualified_name_from_frame",
    "get_globals_from_code",
    "get_last_cache_loading",
    "reset_last_cache_loading",
    "get_function_from_frame",
    "hashing_func_map",
    "user_ignore_files",
]

from .memodisk import (
    DataLoaderWrapper,
    add_data_dependency,
    get_function_from_frame,
    get_function_qualified_name_from_frame,
    get_globals_from_code,
    get_last_cache_loading,
    hashing_func_map,
    memoize,
    open_delay,
    reset_last_cache_loading,
    set_cache_dir,
    user_ignore_files,
)

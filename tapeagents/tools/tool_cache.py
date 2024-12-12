import json
import logging
import os
import threading
from typing import Any, Callable

from termcolor import colored

_CACHE_PATH = "tool_cache.jsonl"
_FORCE_CACHE = False
_cache = {}

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

lock = threading.Lock()


def cached_tool(tool_fn) -> Callable:
    def wrapper(*args, **kwargs):
        fn_name = getattr(tool_fn, "__name__", repr(tool_fn))
        if result := get_from_cache(fn_name, args, kwargs):
            return result
        if _FORCE_CACHE:
            raise ValueError(f"Tool {fn_name} forced cache miss. Tool cache size {len(_cache.get(fn_name, {}))}")
        result = tool_fn(*args, **kwargs)
        add_to_cache(fn_name, args, kwargs, result)
        return result

    return wrapper


def get_from_cache(fn_name: str, args: tuple, kwargs: dict) -> Any:
    if _FORCE_CACHE:
        assert os.path.exists(_CACHE_PATH), f"Cache file {_CACHE_PATH} does not exist"
    if not _cache and os.path.exists(_CACHE_PATH):
        with open(_CACHE_PATH, "r") as f:
            for line in f:
                data = json.loads(line)
                tool_cache = _cache.get(data["fn_name"], {})
                key = json.dumps((data["args"], data["kwargs"]), sort_keys=True)
                tool_cache[key] = data["result"]
                _cache[data["fn_name"]] = tool_cache
    key = json.dumps((args, kwargs), sort_keys=True)
    result = _cache.get(fn_name, {}).get(key)
    if result is not None:
        logger.info(colored(f"Tool cache hit '{fn_name}', args {args}, kwargs {kwargs}", "green"))
    else:
        logger.info(colored(f"Tool cache miss '{fn_name}', args {args}, kwargs {kwargs}", "yellow"))
    return result


def add_to_cache(fn_name: str, args: tuple, kwargs: dict, result: Any):
    logger.info(f"Adding {fn_name} with args {args} and kwargs {kwargs} to cache")
    tool_cache = _cache.get(fn_name, {})
    key = json.dumps((args, kwargs), sort_keys=True)
    tool_cache[key] = result
    _cache[fn_name] = tool_cache
    with lock:
        with open(_CACHE_PATH, "a") as f:
            f.write(json.dumps({"fn_name": fn_name, "args": args, "kwargs": kwargs, "result": result}) + "\n")

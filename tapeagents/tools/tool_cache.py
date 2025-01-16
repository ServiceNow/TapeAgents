import json
import logging
import os
import threading
from typing import Any, Callable

from termcolor import colored

_CACHE_NAME = "tool_cache.jsonl"
_FORCE_CACHE = False
_cache = {}

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


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
    cache_dir = os.getenv("_CACHE_DIR", ".")
    if _FORCE_CACHE:
        cache_file = os.path.join(os.getenv(cache_dir, _CACHE_NAME))
        assert os.path.exists(cache_file), f"Cache {cache_file} does not exist"
    if not _cache and os.path.exists(cache_dir):
        cnt = 0
        for fname in os.listdir(cache_dir):
            if not fname.startswith(_CACHE_NAME):
                continue
            cache_file = os.path.join(os.getenv(cache_dir, fname))
            with open(cache_file) as f:
                for line in f:
                    data = json.loads(line)
                    tool_cache = _cache.get(data["fn_name"], {})
                    key = json.dumps((data["args"], data["kwargs"]), sort_keys=True)
                    tool_cache[key] = data["result"]
                    cnt += 1
                    _cache[data["fn_name"]] = tool_cache
        logger.info(f"Loaded {cnt} tool cache entries from {cache_dir}")
    key = json.dumps((args, kwargs), sort_keys=True)
    result = _cache.get(fn_name, {}).get(key)
    if result is not None:
        logger.info(colored(f"Tool cache hit for {fn_name}", "green"))
    else:
        logger.info(colored(f"Tool cache miss for {fn_name}", "yellow"))
    return result


def add_to_cache(fn_name: str, args: tuple, kwargs: dict, result: Any):
    logger.info(f"Adding {fn_name} with args {args} and kwargs {kwargs} to cache")
    tool_cache = _cache.get(fn_name, {})
    key = json.dumps((args, kwargs), sort_keys=True)
    tool_cache[key] = result
    _cache[fn_name] = tool_cache
    fname = os.path.join(os.getenv("_CACHE_DIR", "."), f"{_CACHE_NAME}.{os.getpid()}.{threading.get_native_id()}")
    with open(fname, "a") as f:
        f.write(json.dumps({"fn_name": fn_name, "args": args, "kwargs": kwargs, "result": result}) + "\n")

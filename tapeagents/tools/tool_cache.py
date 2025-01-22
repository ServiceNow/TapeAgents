import json
import logging
import os
import threading
from typing import Any, Callable

from termcolor import colored

from tapeagents.config import common_cache_dir, force_cache
from tapeagents.utils import FatalError

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

_CACHE_PREFIX = "tool_cache"
_cache = {}


def cached_tool(tool_fn) -> Callable:
    def wrapper(*args, **kwargs):
        fn_name = getattr(tool_fn, "__name__", repr(tool_fn))
        if result := get_from_cache(fn_name, args, kwargs):
            return result
        if force_cache():
            raise FatalError(f"Cache is forced but no cache entry found for {fn_name}({args}, {kwargs})")
        result = tool_fn(*args, **kwargs)
        add_to_cache(fn_name, args, kwargs, result)
        return result

    return wrapper


def get_from_cache(fn_name: str, args: tuple, kwargs: dict) -> Any:
    global _cache
    if not _cache:
        load_cache()
    key = json.dumps((args, kwargs), sort_keys=True)
    result = _cache.get(fn_name, {}).get(key)
    if result is not None:
        logger.info(colored(f"Tool cache hit for {fn_name}", "green"))
    else:
        logger.info(colored(f"Tool cache miss for {fn_name}", "yellow"))
    return result


def load_cache():
    global _cache
    cache_dir = common_cache_dir()
    if os.path.exists(cache_dir):
        for fname in os.listdir(cache_dir):
            if not fname.startswith(_CACHE_PREFIX):
                continue
            with open(os.path.join(cache_dir, fname)) as f:
                for line in f:
                    data = json.loads(line)
                    tool_cache = _cache.get(data["fn_name"], {})
                    key = json.dumps((data["args"], data["kwargs"]), sort_keys=True)
                    tool_cache[key] = data["result"]
                    _cache[data["fn_name"]] = tool_cache
    for k, v in _cache.items():
        logger.info(f"Loaded {len(v)} cache entries for {k}")


def add_to_cache(fn_name: str, args: tuple, kwargs: dict, result: Any):
    global _cache
    logger.info(f"Adding {fn_name} with args {args} and kwargs {kwargs} to cache")
    tool_cache = _cache.get(fn_name, {})
    key = json.dumps((args, kwargs), sort_keys=True)
    tool_cache[key] = result
    _cache[fn_name] = tool_cache
    fname = os.path.join(
        common_cache_dir(), f"{_CACHE_PREFIX}.{fn_name}.{os.getpid()}.{threading.get_native_id()}.jsonl"
    )
    with open(fname, "a") as f:
        f.write(json.dumps({"fn_name": fn_name, "args": args, "kwargs": kwargs, "result": result}) + "\n")

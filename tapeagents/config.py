import os

DB_DEFAULT_FILENAME = "tapedata.sqlite"
ATTACHMENT_DEFAULT_DIR = "attachments"


def common_cache_dir():
    return os.getenv("TAPEAGENTS_CACHE_DIR", ".cache")


def force_cache():
    return bool(os.environ.get("TAPEAGENTS_FORCE_CACHE", 0))


def is_debug_mode():
    return os.environ.get("TAPEAGENTS_DEBUG", None) == "1"


def sqlite_db_path():
    return os.environ.get("TAPEAGENTS_SQLITE_DB", DB_DEFAULT_FILENAME)

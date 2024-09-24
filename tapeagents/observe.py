import datetime
import json
import logging
import sqlite3
from typing import Callable, Type

from pydantic import BaseModel

from .config import sqlite_db_path
from .core import LLMCall, LLMMessage, Prompt, Tape

logger = logging.getLogger(__name__)

_checked_sqlite = False


LLMCallListener = Callable[[LLMCall], None]
TapeListener = Callable[[Tape], None]


def init_sqlite_if_not_exists(only_once: bool = False):
    """
    Ensure that the tables exist in the sqlite database.

    This is only done once per Python process.
    If you want to change the SQLite path during at run time, you can run this function manually
    with only_once=False.

    """
    global _checked_sqlite
    if _checked_sqlite and only_once:
        return

    path = sqlite_db_path()
    logger.info(f"use SQLite db at {path}")
    conn = sqlite3.connect(path)
    cursor = conn.cursor()
    cursor.execute("""               
    CREATE TABLE IF NOT EXISTS LLMCalls (
        prompt_id TEXT PRIMARY KEY,
        timestamp TEXT,
        prompt TEXT,
        completion TEXT,
        prompt_length_tokens INTEGER,
        completion_length_tokens INTEGER,
        cached INTEGER
    )
    """)
    # now create tape table with tape_id index and data column
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS Tapes (
        tape_id TEXT PRIMARY KEY,
        timestamp TEXT,
        length INTEGER,
        metadata TEXT,
        context TEXT,
        steps TEXT
    )
    """)
    cursor.close()
    _checked_sqlite = True


def sqlite_store_llm_call(call: LLMCall):
    init_sqlite_if_not_exists()
    with sqlite3.connect(sqlite_db_path()) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO LLMCalls (prompt_id, timestamp, prompt, completion, prompt_length_tokens, completion_length_tokens, cached) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                call.prompt.id,
                call.timestamp,
                call.prompt.model_dump_json(),
                call.completion.model_dump_json(),
                call.prompt_length_tokens,
                call.completion_length_tokens,
                call.cached,
            ),
        )
        cursor.close()


def sqlite_store_tape(tape: Tape):
    init_sqlite_if_not_exists()
    with sqlite3.connect(sqlite_db_path()) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO Tapes (tape_id, timestamp, length, metadata, context, steps) VALUES (?, ?, ?, ?, ?, ?)",
            (
                tape.metadata.id,
                datetime.datetime.now().isoformat(),
                len(tape),
                tape.metadata.model_dump_json(),
                tape.context.model_dump_json() if isinstance(tape.context, BaseModel) else json.dumps(tape.context),
                json.dumps([step.model_dump() for step in tape.steps]),
            ),
        )
        cursor.close()


llm_call_listeners: list[LLMCallListener] = [sqlite_store_llm_call]
tape_listeners: list[TapeListener] = [sqlite_store_tape]


def observe_llm_call(call: LLMCall):
    for listener in llm_call_listeners:
        listener(call)


def retrieve_llm_call(prompt_id: str) -> LLMCall | None:
    init_sqlite_if_not_exists()
    with sqlite3.connect(sqlite_db_path()) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM LLMCalls WHERE prompt_id = ?", (prompt_id,))
        row = cursor.fetchone()
        cursor.close()
        if row is None:
            return None
        # ignore row[0] cause it is alredy in row[2]
        return LLMCall(
            timestamp=row[1],
            prompt=Prompt.model_validate_json(row[2]),
            completion=LLMMessage.model_validate_json(row[3]),
            prompt_length_tokens=row[4],
            completion_length_tokens=row[5],
            cached=row[6],
        )


def observe_tape(tape: Tape):
    for listener in tape_listeners:
        listener(tape)


def retrieve_tape_llm_calls(tape: Tape) -> dict[str, LLMCall]:
    result = {}
    for step in tape:
        if prompt_id := getattr(step, "prompt_id", step.metadata.prompt_id):
            if call := retrieve_llm_call(prompt_id):
                result[prompt_id] = call
    return result


def retrieve_tape(tape_class: Type, tape_id: str) -> Tape:
    init_sqlite_if_not_exists()
    with sqlite3.connect(sqlite_db_path()) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM Tapes WHERE tape_id = ?", (tape_id,))
        row = cursor.fetchone()
        cursor.close()
        if row is None:
            raise ValueError(f"No tape found with id {tape_id}")
        return tape_class.model_validate(
            dict(
                metadata=json.loads(row[3]),
                context=json.loads(row[4]),
                steps=json.loads(row[5]),
            )
        )


def get_latest_tape_id() -> str:
    init_sqlite_if_not_exists()
    with sqlite3.connect(sqlite_db_path()) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT tape_id FROM Tapes ORDER BY timestamp DESC LIMIT 1")
        row = cursor.fetchone()
        cursor.close()
        if row is None:
            return ""
        return row[0]


def retrieve_all_llm_calls(sqlite_fpath: str) -> list[LLMCall]:
    conn = sqlite3.connect(sqlite_fpath)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT timestamp, prompt, completion, prompt_length_tokens, completion_length_tokens, cached FROM LLMCalls"
    )
    rows = cursor.fetchall()
    cursor.close()
    calls: list[LLMCall] = []
    for timestamp, prompt_str, completion, prompt_length_tokens, completion_length_tokens, cached in rows:
        calls.append(
            LLMCall(
                timestamp=timestamp,
                prompt=Prompt(**json.loads(prompt_str)),
                completion=LLMMessage(**json.loads(completion)),
                prompt_length_tokens=prompt_length_tokens,
                completion_length_tokens=completion_length_tokens,
                cached=cached,
            )
        )
    return calls

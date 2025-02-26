"""
Functions to observe and store LLM calls and Tapes in a persistent storage
"""

import datetime
import json
import logging
import queue
import sqlite3
import threading
import time
from typing import Callable, Optional, Type

from pydantic import BaseModel

from tapeagents.config import sqlite_db_path
from tapeagents.core import LLMCall, LLMOutput, Prompt, Tape

logger = logging.getLogger(__name__)

_checked_sqlite = False
_WRITER_THREAD: Optional["SQLiteWriterThread"] = None

LLMCallListener = Callable[[LLMCall], None]
TapeListener = Callable[[Tape], None]


def init_sqlite_if_not_exists(only_once: bool = True):
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
        output TEXT,
        prompt_length_tokens INTEGER,
        output_length_tokens INTEGER,
        cached INTEGER,
        llm_info TEXT,
        cost REAL
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


def erase_sqlite():
    path = sqlite_db_path()
    conn = sqlite3.connect(path)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM LLMCalls")
    cursor.execute("DELETE FROM Tapes")
    cursor.close()
    conn.commit()  # Don't forget to commit the changes
    conn.close()  # Close the connection when done


def _do_sqlite3_store_llm_call(call: LLMCall):
    try:
        init_sqlite_if_not_exists()
        with sqlite3.connect(sqlite_db_path(), timeout=30) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO LLMCalls (prompt_id, timestamp, prompt, output, prompt_length_tokens, output_length_tokens, cached, llm_info, cost) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    call.prompt.id,
                    call.timestamp,
                    call.prompt.model_dump_json(),
                    call.output.model_dump_json(),
                    call.prompt_length_tokens,
                    call.output_length_tokens,
                    call.cached,
                    json.dumps(call.llm_info),
                    call.cost,
                ),
            )
            cursor.close()
    except Exception as e:
        logger.error(f"Failed to store LLMCall: {e}")


def sqlite_store_llm_call(call: LLMCall):
    """Standalone function to store LLM calls.

    Will use the queue if available (within context manager),
    otherwise falls back to single-threaded mode.
    """
    if _WRITER_THREAD is not None and _WRITER_THREAD.queue is not None:
        # We're in a context manager, use the queue
        logger.debug("Using SQLite queue writing mode")
        _WRITER_THREAD.queue.put(call)
    else:
        # We're not in a context manager, use single-threaded mode
        logger.debug("Using single-threaded SQLite writing mode")
        _do_sqlite3_store_llm_call(call)


def sqlite_store_tape(tape: Tape):
    try:
        init_sqlite_if_not_exists()
        with sqlite3.connect(sqlite_db_path(), timeout=30) as conn:
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
    except Exception as e:
        logger.error(f"Failed to store Tape: {e}")


llm_call_listeners: list[LLMCallListener] = [sqlite_store_llm_call]
tape_listeners: list[TapeListener] = [sqlite_store_tape]


def observe_llm_call(call: LLMCall):
    for listener in llm_call_listeners:
        listener(call)


def retrieve_llm_calls(prompt_ids: str | list[str]) -> list[LLMCall]:
    if isinstance(prompt_ids, str):
        prompt_ids = [prompt_ids]
    init_sqlite_if_not_exists()
    llm_calls = []
    with sqlite3.connect(sqlite_db_path()) as conn:
        cursor = conn.cursor()
        for i in range(0, len(prompt_ids), 100):
            prompts = prompt_ids[i : i + 100]
            cursor.execute(
                f"SELECT * FROM LLMCalls WHERE prompt_id IN ({','.join(['?'] * len(prompts))})",
                prompts,
            )
            rows = cursor.fetchall()
            for row in rows:
                # ignore row[0] cause it is alredy in row[2]
                llm_calls.append(
                    LLMCall(
                        timestamp=row[1],
                        prompt=Prompt.model_validate_json(row[2]),
                        output=LLMOutput.model_validate_json(row[3]),
                        prompt_length_tokens=row[4],
                        output_length_tokens=row[5],
                        cached=row[6],
                        llm_info=json.loads(row[7]),
                        cost=row[8],
                    )
                )
        cursor.close()
    return llm_calls


def observe_tape(tape: Tape):
    for listener in tape_listeners:
        listener(tape)


def retrieve_tape_llm_calls(tapes: Tape | list[Tape]) -> dict[str, LLMCall]:
    logger.info("Retrieving LLM calls")
    if isinstance(tapes, Tape):
        tapes = [tapes]
    result = {}
    prompt_ids = list(set([step.metadata.prompt_id for tape in tapes for step in tape if step.metadata.prompt_id]))
    result = {call.prompt.id: call for call in retrieve_llm_calls(prompt_ids)}
    logger.info(f"Retrieved {len(result)} LLM calls")
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


def retrieve_all_llm_calls(sqlite_fpath: str | None = None) -> list[LLMCall]:
    sqlite_fpath = sqlite_fpath or sqlite_db_path()
    init_sqlite_if_not_exists()
    conn = sqlite3.connect(sqlite_fpath)

    def dict_factory(cursor, row):
        d = {}
        for idx, col in enumerate(cursor.description):
            d[col[0]] = row[idx]
        return d

    conn.row_factory = dict_factory
    cursor = conn.cursor()
    cursor.execute(
        "SELECT timestamp, prompt, output, prompt_length_tokens, output_length_tokens, cached, llm_info, cost FROM LLMCalls"
    )
    rows = cursor.fetchall()
    cursor.close()
    calls: list[LLMCall] = []
    for row in rows:
        calls.append(
            LLMCall(
                timestamp=row["timestamp"],
                prompt=Prompt(**json.loads(row["prompt"])),
                output=LLMOutput(**json.loads(row["output"])),
                prompt_length_tokens=row["prompt_length_tokens"],
                output_length_tokens=row["output_length_tokens"],
                cached=row["cached"],
                llm_info=json.loads(row["llm_info"]),
                cost=row["cost"],
            )
        )
    return calls


class SQLiteWriterThread:
    def __init__(self):
        self.write_queue: Optional[queue.Queue] = None
        self.writer_thread: Optional[threading.Thread] = None

    def __enter__(self):
        """Start the SQLite queue writer when entering the context."""
        if self.write_queue is not None:
            return self  # Already running

        self.write_queue = queue.Queue()
        self.writer_thread = threading.Thread(target=self._queue_sqlite_writer, daemon=True)
        self.writer_thread.start()

        # Set the global reference
        global _WRITER_THREAD
        _WRITER_THREAD = self
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop the SQLite queue writer when exiting the context."""
        if self.write_queue is not None and self.writer_thread is not None:
            self.wait_for_empty()
            self.write_queue.put(None)  # Signal thread to stop
            self.writer_thread.join()  # Wait for thread to finish
            self.write_queue = None
            self.writer_thread = None

            # Clear the global reference
            global _WRITER_THREAD
            _WRITER_THREAD = None

    def _queue_sqlite_writer(self):
        """The worker function that processes the queue."""
        while True:
            item = self.write_queue.get()
            if item is None:  # Stop signal
                break
            _do_sqlite3_store_llm_call(item)
            self.write_queue.task_done()

    def wait_for_empty(self, timeout: Optional[float] = None) -> bool:
        """Wait for the queue to be empty and all tasks to be processed."""
        if self.write_queue is None:
            return True

        try:
            self.write_queue.join()
            start_time = time.monotonic()
            logger.info("Waiting for SQLite queue to empty...")
            while not self.write_queue.empty():
                if timeout is not None:
                    elapsed = time.monotonic() - start_time
                    if elapsed >= timeout:
                        return False
                time.sleep(0.1)
                self.write_queue.join()
            return True
        except Exception as e:
            logger.error(f"Error while waiting for queue to empty: {e}")
            return False

    @property
    def queue(self) -> Optional[queue.Queue]:
        """Access the write queue."""
        return self.write_queue

    @property
    def is_empty(self) -> bool:
        """Check if the queue is empty."""
        return self.write_queue is None or self.write_queue.empty()

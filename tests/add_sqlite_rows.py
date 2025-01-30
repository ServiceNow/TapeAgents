import os
import sqlite3
import sys

_llm_calls = "LLMCalls"
_db = "tapedata.sqlite"


def add_rows_from_db(source_db_path, target_db_path):
    source_conn = sqlite3.connect(source_db_path)
    target_conn = sqlite3.connect(target_db_path)

    source_cursor = source_conn.cursor()
    target_cursor = target_conn.cursor()

    source_cursor.execute(f"SELECT * FROM {_llm_calls}")
    rows = source_cursor.fetchall()

    for row in rows:
        placeholders = ", ".join(["?"] * len(row))
        target_cursor.execute(f"INSERT INTO {_llm_calls} VALUES ({placeholders})", row)

    target_conn.commit()
    source_conn.close()
    target_conn.close()


def main():
    dirname = sys.argv[1]
    db_source = sys.argv[2]
    for root, dirs, files in os.walk(dirname):
        for file in files:
            if file.endswith(_db):
                db_path = os.path.join(root, file)
                print(db_path)
                add_rows_from_db(db_source, db_path)


if __name__ == "__main__":
    assert len(sys.argv) == 3, "Usage: python -m tests.add_sqlite_rows <dirname> <db_source>"
    main()

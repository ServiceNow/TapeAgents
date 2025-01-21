import os
import sqlite3
import sys

_llm_calls = "LLMCalls"


def add_columns_to_db(db_path):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute(f"PRAGMA table_info{_llm_calls}")
    columns = [row[1] for row in c.fetchall()]
    if "llm_info" not in columns:
        c.execute(f"ALTER TABLE {_llm_calls} ADD COLUMN llm_info TEXT DEFAULT '{{}}'")
    if "cost" not in columns:
        c.execute(f"ALTER TABLE {_llm_calls} ADD COLUMN cost REAL DEFAULT 0.0")
    conn.commit()
    conn.close()


def main():
    dirname = sys.argv[1]
    for root, dirs, files in os.walk(dirname):
        for file in files:
            if file.endswith(".sqlite"):
                db_path = os.path.join(root, file)
                print(db_path)
                add_columns_to_db(db_path)


if __name__ == "__main__":
    assert len(sys.argv) == 2, "Usage: python -m tests.add_sqlite_columns <dirname>"
    main()

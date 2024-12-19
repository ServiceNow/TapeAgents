import gzip
import os
import shutil
import sys

_db = "tapedata.sqlite"


def compress_file(file_path):
    with open(file_path, "rb") as f_in:
        with gzip.open(file_path + ".gz", "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
    # os.remove(file_path)


def main():
    dirname = sys.argv[1]
    for root, dirs, files in os.walk(dirname):
        for file in files:
            if file.endswith(_db):
                file_path = os.path.join(root, file)
                print(f"Compressing {file_path}")
                compress_file(file_path)


if __name__ == "__main__":
    assert len(sys.argv) == 2, "Usage: python compress_sqlite.py <dirname>"
    main()

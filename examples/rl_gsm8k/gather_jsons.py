# read json files from a folder, create new json with the same name that contains all the content

import json
import os
import sys


def gather_jsons(folder: str):
    all_jsons = []
    for root, _, files in os.walk(folder):
        for file in files:
            if file.endswith(".json"):
                with open(os.path.join(root, file)) as f:
                    all_jsons.append(json.load(f))

    dst_dir = f"{folder}/all"
    os.makedirs(dst_dir, exist_ok=True)
    dst_name = f"{dst_dir}/tapes.json"
    with open(dst_name, "w") as f:
        json.dump(all_jsons, f, indent=4)


gather_jsons(sys.argv[1])

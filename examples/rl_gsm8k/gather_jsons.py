# read json files from a folder, create new json with the same name that contains all the content

import json
import os
from pathlib import Path

def gather_jsons(folder: str):
    all_jsons = []
    for root, _, files in os.walk(folder):
        for file in files:
            if file.endswith(".json"):
                with open(os.path.join(root, file)) as f:
                    all_jsons.append(json.load(f))
    
    dst_name = f"{folder}.json"
    with open(dst_name, "w") as f:
        json.dump(all_jsons, f, indent=4)


gather_jsons("outputs/rl_debug_v6/tapes/train/0")
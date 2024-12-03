# DELETEME before the release

import json
from pathlib import Path
from tqdm import tqdm
import yaml

from llmd2.tapeagents_tmp.ghreat.critic import from_llmd2_annotated_dialog_dict
import llmd2.tapeagents_tmp.ghreat.tape as tape_module
from llmd2.tapeagents_tmp.ghreat.tape import from_llmd2_dialog_dict
from tapeagents.io import stream_yaml_tapes


def main():
    # src = "/mnt/llmd/data/dzmitry/user_forks/test_rewrite_forks/data_last10.yaml"
    # dst = "/mnt/llmd/data/dzmitry/ghreat_tapes/tmp_last10.yaml"
    
    # src = "/mnt/llmd/data/dzmitry/user_forks/test_rewrite_forks/data.yaml"
    # dst = "/mnt/llmd/data/dzmitry/ghreat_tapes/test_rewrite_forks_data.yaml"
    
    # src = "/mnt/llmd/data/ehsan_kamalloo/make_dialogue_tree/FlyCorp/tree_config2/multistep_nico_teacher_prompt_v4/v1/size50/agent_forks/data.yaml"
    # dst = "/mnt/llmd/data/dzmitry/ghreat_tapes/agent_forks.yaml"

    # CoffeeCorpGoldV2 annotations
    # src = "/mnt/llmd/data/gabriel/make_annotation_traces/formfiller_v2_eval100/AllTrain/agent_all_metrics_traces/rebalance_forks/dialogues_forked.yaml"
    # dst = "/mnt/llmd/data/gontiern/tapes/CoffeeCorpGoldV2_annotations.yaml"

    # formfiller_v3 annotations
    # src = "/mnt/llmd/data/gabriel/make_annotation_traces/formfiller_v3/data_balanced.yaml"
    # dst = "/mnt/llmd/data/gontiern/tapes/formfiller_v3_annotations.yaml"

    # balanced_ufs_part0 annotations
    src = "/mnt/llmd/data/gontiern/make_balanced_set/ehsan_balanced_v3/balanced_ufs_part2.yaml"
    dst = "/mnt/llmd/data/ehsan_kamalloo/tapes/make_balanced_set/ehsan_balanced_v3/balanced_ufs_part2.yaml"

    dialogs = list(yaml.safe_load_all(open(src)))
    tapes = [from_llmd2_dialog_dict(d) for d in tqdm(dialogs, desc="Converting tapes")]
    # tapes = [from_llmd2_annotated_dialog_dict(d) for d in tqdm(dialogs, desc="Converting tapes")]
    with stream_yaml_tapes(Path(dst)) as saver: 
        for tape in tqdm(tapes, desc="Saving tapes"):
            saver.save(tape)


if __name__ == "__main__":
    main()
import json
from pathlib import Path
import yaml

from examples.form_filler.tape import from_llmd2_dialog_dict, FormFillerTape, FormFillerStep
from examples.form_filler.state import FormFillerState, compute_form_filler_state
from tapeagents.io import save_tapes, load_tapes


def main():
    src_path = "/mnt/llmd/data/dzmitry/ghreat_tapes/test_rewrite_forks_data.yaml"
    gt_states_path = "/mnt/llmd/data/dzmitry/ghreat_tapes/test_rewrite_forks_states.json"
    tapes = load_tapes(FormFillerTape, src_path)
    gt_states = json.load(open(gt_states_path))
    assert len(tapes) == len(gt_states)
    
    for i, (tape, gt_state) in enumerate(zip(tapes, gt_states)):
        state = compute_form_filler_state(tape)
        state = state.model_dump()
        
        # remove minor mismatches
        del gt_state["last_user_text_message"]
        skipped = gt_state["function_parameters_skipped"]
        if skipped and not any(skipped.values()):
            gt_state["function_parameters_skipped"] = {}
        
        # check if keys are the same
        if state.keys() != gt_state.keys():
            print(f"Keys mismatch in tape {i}")
            print(state.keys())
            print(gt_state.keys())
            break
        
        mismatch = False
        for key in state.keys():
            if state[key] != gt_state[key]:
                print(f"Mismatch in tape {i}, key {key}")
                gt_dump = json.dumps(gt_state[key], indent=2)
                dump = json.dumps(state[key], indent=2)
                print("llmd", gt_dump)
                print("tapeagents", dump)
                # print diff between dumps using difflib
                import difflib
                diff = difflib.unified_diff(gt_dump.splitlines(), dump.splitlines())
                print("\n".join(diff))
                
                mismatch = True
                break
        if mismatch:
            break
            
    
    
if __name__ == "__main__":
    main()
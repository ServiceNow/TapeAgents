import json
from llmd2.core.trajectory import to_trajectory
from scripts.file_tools import load_function_dialogues


src_path = "/mnt/llmd/data/dzmitry/user_forks/test_rewrite_forks/data.yaml"
dst_path = "/mnt/llmd/data/dzmitry/ghreat_tapes/test_rewrite_forks_states.json"
dialogues = load_function_dialogues(src_path)
trajectories = [to_trajectory(d) for d in dialogues]
states = [t.last_state.model_dump() for t in trajectories]
with open(dst_path, "w") as f:
    json.dump(states, f, indent=2)


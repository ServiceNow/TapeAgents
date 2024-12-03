from collections import defaultdict
from pathlib import Path
import yaml
import pandas as pd

# path = Path('/mnt/llmd/data/gabriel/make_tape_tree/train/BigBankCorp/agent_teacher_agent_vllm_llama3_8b_temp1/user_vllm_llama3_8b_temp1_tree_config5/size_10')
path = Path('/mnt/llmd/data/ehsan_kamalloo/make_tape_tree/teacher')
# Store results for creating a DataFrame
all_counts = []

for i in range(0, 100, 2):
    layer_dir = path / f'layer_{i}'
    if not layer_dir.exists():
        print(f'Layer {i} does not exist')
        break
    user_simulator_path = layer_dir / 'user_simulator_tapes.yaml'
    with open(user_simulator_path) as fp:
        user_simulator_tapes = list(yaml.safe_load_all(fp))
    counts = defaultdict(int)
    for tape in user_simulator_tapes:
        if tape['steps']:
            key = tape['steps'][-1]['new_observation']['metadata']['other']['alias']
            counts[key] += 1
        else:
            counts['FAILURE'] += 1
    counts = dict(counts)
    print(f'Layer {i}: {counts}')
    
    # Append counts to the result list for DataFrame
    for key, value in counts.items():
        all_counts.append({'Layer': i, 'Alias': key, 'Count': value})

# Convert to DataFrame
df = pd.DataFrame(all_counts)

# Print DataFrame
print(df)

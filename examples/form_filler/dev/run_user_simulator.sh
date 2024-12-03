#!/bin/bash

# Configurable output path prefix
OUTPUT_PREFIX=".local/debug/user_tapes"

# List of user simulator agents
USER_SIMULATORS=(
  ask_about_docs
  happy_path
  init_message_short
  invalid_value_3
  skip_unskipable
  init_message_ask
  invalid_instruct
  multislot_instruct3
  skip_optional
)

# Loop over each user simulator agent
for USER_SIMULATOR in "${USER_SIMULATORS[@]}"; do
  # Run the Python script with the specified arguments
  python llmd2-core/src/llmd2/tapeagents_tmp/ghreat/dev/run_user_simulator.py \
    input_dialogues_path="/mnt/llmd/data/gontiern/tapes/ehsan_balanced_v3_balanced_ufs_part0/teacher_predicted_tapes.yaml" \
    output_path="${OUTPUT_PREFIX}/${USER_SIMULATOR}" \
    user_simulator_agent="${USER_SIMULATOR}" \
    llm@user_simulator_agent.llms=vllm_llama3_8b \
    max_continuable_tapes=30 \
    n_workers=4
done

# Visualize 
python llmd2-core/src/llmd2/tapeagents_tmp/ghreat/dev/visualize_formfiller_tapes.py ${OUTPUT_PREFIX}
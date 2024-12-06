#!/bin/bash

INPUT_PATH=".local/user_tapes/happy_path/user_simulator_tapes.yaml"
OUTPUT_PATH=".local/agent_tapes"

AGENT=teacher_agent  # teacher_agent | student_agent

python examples/form_filler/dev/run_formfiller_agent.py \
  agent="${AGENT}" \
  user_dialogues_path="${INPUT_PATH}" \
  output_path="${OUTPUT_PATH}" \
  n_workers=6

# Visualize 
python examples/form_filler/dev/visualize_formfiller_tapes.py ${OUTPUT_PATH}

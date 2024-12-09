#!/bin/bash

INPUT_PATH="examples/form_filler/assets/initial_user_tapes.yaml"
AGENT=teacher_agent  # teacher_agent | student_agent
OUTPUT_PATH="outputs/agent_tapes/${AGENT}"

python examples/form_filler/scripts/run_formfiller_agent.py \
  agent="${AGENT}" \
  user_dialogues_path="${INPUT_PATH}" \
  output_path="${OUTPUT_PATH}" \
  n_workers=6

# Visualize 
python examples/form_filler/scripts/tape_browser.py ${OUTPUT_PATH}

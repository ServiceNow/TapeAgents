#!/bin/bash

# output path
OUTPUT_PREFIX=".local/debug/agent_tapes"

# agent type
# AGENT=teacher_agent
AGENT=student_agent

python examples/form_filler/dev/run_formfiller_agent.py \
  agent="${AGENT}" \
  user_dialogues_path=/mnt/llmd/data/ehsan_kamalloo/tapes/make_balanced_set/ehsan_balanced_v3/balanced_ufs_part0.yaml \
  output_path="${OUTPUT_PREFIX}" \
  n_workers=6

# Visualize 
python examples/form_filler/dev/visualize_formfiller_tapes.py ${OUTPUT_PREFIX}
#!/bin/bash

SIZE=5
TREE_CONFIG=tree_config6
AGENT="teacher_agent"
AGENT_LLM="vllm_llama3_405b_temp1"
USER_LLM="vllm_llama3_405b_temp1"


DOMAIN="train/FlyCorp"
# DOMAIN="train/CoffeeCorp"
# DOMAIN="train/BigBankCorp"

OUTPUT_DIR=.local/make_tape_tree/${DOMAIN}/${AGENT}_${AGENT_LLM}/user_${USER_LLM}/${TREE_CONFIG}_size${SIZE}
echo "Output directory: ${OUTPUT_DIR}"

JOB_NAME=$(echo "tapetree_${DOMAIN}_${SIZE}" | tr '[:upper:]' '[:lower:]' | sed 's/\///g' | tr '-' '_' | tr -d '.')


uv run -m examples.form_filler.scripts.make_tape_tree \
    make_tape_tree.force_restart_idx=0 \
    make_tape_tree.output_path=${OUTPUT_DIR} \
    make_tape_tree.preambles_path=examples/form_filler/assets/forms/${DOMAIN} \
    make_tape_tree.global_count=${SIZE} \
    make_tape_tree.user_simulator_llm=${USER_LLM} \
    make_tape_tree.num_workers=6 \
    agent@make_tape_tree.agent=${AGENT} \
    llm@make_tape_tree.agent.llm=${AGENT_LLM}
#!/bin/bash

SIZE=500
TREE_CONFIG=tree_config6
AGENT="teacher_agent"
AGENT_LLM="vllm_llama3_405b_temp1"
USER_LLM="vllm_llama3_405b_temp1"

# AGENT_LLM="vllm_llama3_8b_temp1"
# USER_LLM="vllm_llama3_8b_temp1"


DOMAIN="train/FlyCorp"
# DOMAIN="train/CoffeeCorp"
# DOMAIN="train/BigBankCorp"

OUTPUT_DIR=/mnt/llmd/data/gontiern/make_tape_tree/${DOMAIN}/agent_${AGENT}_${AGENT_LLM}/user_${USER_LLM}/${TREE_CONFIG}_size${SIZE}
echo "Output directory: ${OUTPUT_DIR}"

JOB_NAME=$(echo "tapetree_${DOMAIN}_${SIZE}" | tr '[:upper:]' '[:lower:]' | sed 's/\///g' | tr '-' '_' | tr -d '.')


python llmd2-core/src/llmd2/tapeagents_tmp/ghreat/dev/make_tape_tree.py \
    make_tape_tree.force_restart_idx=0 \
    make_tape_tree.output_path=${OUTPUT_DIR} \
    make_tape_tree.preambles_path=/mnt/llmd/data/gontiern/catalog_items/for_open_release/${DOMAIN} \
    make_tape_tree.global_count=${SIZE} \
    make_tape_tree.user_simulator_llm=${USER_LLM} \
    make_tape_tree.num_workers=6 \
    agent@make_tape_tree.agent=${AGENT} \
    llm@make_tape_tree.agent.llm=${AGENT_LLM}

# make job \
#     JOB_ACCOUNT=snow.research.tapes \
#     GPU_MEM=0 CPU=8 CPU_MEM=128 GPU=0 JOB_NAME="${JOB_NAME}" \
#     ACCELERATE=0 SNAPSHOT=1 DRY_RUN=0 LOCAL=0 NPROC=1 TOOLKIT_PROFILE=yul201 CONDA=1 \
#     COMMAND="${COMMAND}"

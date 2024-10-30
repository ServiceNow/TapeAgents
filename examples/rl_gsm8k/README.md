# RL for GSM8k (WIP)

This example demonstrates how to train a reinforcement learning agent to solve math problems from the [GSM8k](https://huggingface.co/datasets/openai/gsm8k) dataset.

## Quickstart

The example can be run with the following command on a H100s (should also run on a A100):

```bash
python -m examples.rl_gsm8k.orchestrate_rl
```

## Overview

![image](https://github.com/user-attachments/assets/c715de7a-8d15-4504-9c7c-d8ad28726941)

### Collect online RL training data

#### Collect tapes
* the current model (updated llama 3.1 8b) is served on all the gpus using vllm. 
* a subset of 16 tasks from the train set of gsm8k is sampled and replicated 64 times each for a total of 1024 tasks. 
* the agent produce complete tapes for each of these 1024 tasks using temperature 0.7. 
* traces are created from these new tapes. 
* the log prob of the traces under the current model are computed.

#### Annotate tapes with rewards
* For each trace, the reward is computed as follows:
    * +1 for correct answer
    * 0 for incorrect answer or no answer
    * -1 for step that cannot be parsed to json


#### Annotate tapes with ref log probs
* the current model is taken down and the reference model is now served on all gpus using vllm. 
* the log prob from the reference model (llama 3.1 8b) are computed for the most recent traces. 

### Evaluation (Not shown in the figure)
* every 5 iterations, the agent is evaluated with temperature 0 on the complete test set of gsm8k. No traces are produced on the test set. 

### Finetune
* RL (grpo or reinforce) training is performed on the latest batch of data using a separate process. This makes it easier to manage memory with the main process. Otherwise there are sometimes issues with the gpus not being completely empty after fine tuning. 
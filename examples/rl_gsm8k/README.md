# RL for GSM8k

![alt text](<Screenshot 2025-02-14 at 9.26.34 AM.png>)

This example demonstrates how to train a reinforcement learning [Llama 3.2 1B Instruct](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct) to solve math problems from the [GSM8k](https://huggingface.co/datasets/openai/gsm8k) dataset.

Meta reports that the model obtains [44.4 on the GSM8k dataset (8 shot, CoT)](https://ai.meta.com/blog/llama-3-2-connect-2024-vision-edge-mobile-devices/). In this example, we will be strict and instruct the model to output its answer within `\boxed{}` which it initially fails to do. At the end of training the model respects the format and obtains a score of ~53 on the test set. 

## Quickstart

#### Prerequisities

We use VLLM for inference in our training pipeline. Install it as follows:

```bash
pip install 'tapeagents[finetune]'
```

Make sure you have a Huggingface account with access to <https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct> and use `huggingface-cli` to login to the Hugging Face Hub. You may also want to test your vllm installation and downloading the model's weights by running the following command:

```bash
uv run -m vllm.entrypoints.openai.api_server --model meta-llama/Llama-3.2-1B-Instruct --dtype bfloat16
```

#### Run training loop

The example can be run with the following command on a H100s (should also run on a A100):

```bash
uv run -m examples.rl_gsm8k.orchestrate_rl
```

## Overview

![image](https://github.com/user-attachments/assets/c715de7a-8d15-4504-9c7c-d8ad28726941)

### Collect online RL training data

#### Collect tapes

* the current model is served on the gpus using vllm.
* a subset of 16 tasks from the train set of gsm8k is sampled and replicated 64 times each for a total of 1024 tasks.
* the agent produce complete tapes for each of these 1024 tasks using temperature 0.7.
* traces are created from these new tapes.
* the log prob of the traces under the current model are computed.

#### Annotate tapes with rewards

* For each trace, the reward is computed as follows:
  * +1 for correct answer.
  * 0 for incorrect answer.
  * -1 for step that cannot be parsed to

#### Annotate tapes with ref log probs

* the current model is taken down and the reference model is now served on all gpus using vllm.
* the log prob from the reference model (llama 3.1 8b) are computed for the most recent traces.

### Evaluation (Not shown in the figure)

* every 5 iterations, the agent is evaluated with temperature 0 on the complete test set of gsm8k. No traces are produced on the test set.

### Finetune

* RL (grpo or reinforce) training is performed on the latest batch of data using a separate process. This makes it easier to manage memory with the main process. Otherwise there are sometimes issues with the gpus not being completely empty after fine tuning.

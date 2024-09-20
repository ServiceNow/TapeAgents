# Gaia Agent
The Gaia Agent is an agent that solves the tasks from the [GAIA Benchmark](https://huggingface.co/spaces/gaia-benchmark/leaderboard).
The agent can be backed by OpenAI models or LLAMA models to solve these tasks.

## Requirements
The agent has a special set of requirements that are listed in the `requirements.txt` file in the current folder. Install them with the following command:
```
pip install -r requirements.txt
```

## Quickstart
- `python -m examples.gaia_agent.scripts.evaluate` - script to run evaluation on the validation set.
- `python -m examples.gaia_agent.scripts.tape_browser` - Gradio UI to explore the tapes and metrics produced during evaluation. 

 Dataset location: `gaia-benchmark/GAIA`
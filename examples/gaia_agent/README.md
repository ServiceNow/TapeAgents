# Gaia Agent
The Gaia Agent is an agent that can answer knowledge-grounded questions using web search, calculations and reasoning. The agent can use OpenAI or big LLAMA models to solve these tasks. We demonstrate how it solves the tasks from the [GAIA Benchmark](https://huggingface.co/spaces/gaia-benchmark/leaderboard).

Tape example:  
<img width="867" alt="gaia_perfect_demo_tape" src="https://github.com/user-attachments/assets/a81c22d8-9cf5-42c4-a390-933108753966">



## Structure
The agent is built on top of the [MonoAgent](../../tapeagents/mono_agent.py) class, which implements the following workflow:
- Expose the set of all available actions and thoughts to the model in each prompt
- Render the whole tape into the prompt, trimming only in case when the tape does not fit into the context window
- Based on the end of the current tape, select the short textual guidance prompt that briefly instructs the LLM what to do next
- Append the hints about formatting to the end of the prompt.

The agent is free to choose which thoughts and actions to use to satisfy the current guidance recommendations without the additional constraints of specific node or subagent.

Additionally, the Gaia agent implements the initial planning step, which produces a «plan» in the form of a sequence of free-form descriptions of the actions that should be taken to solve the task.

## Quickstart
- `python -m examples.gaia_agent.scripts.demo` - Interactive Gradio Demo that allows you to set the task for the agent and observe how it solves it step by step.
- `python -m examples.gaia_agent.scripts.evaluate` - script to run evaluation on the GAIA validation set.
- `python -m examples.gaia_agent.scripts.tape_browser` - Gradio UI for exploring the tapes and metrics produced during evaluation.

## Results
| Model | Avg. Val Accuracy | Val Level 1 Accuracy|  Val Level 2 Accuracy |  Val Level 3 Accuracy |
| --- | --- |  --- | --- | --- |
| gpt-4o maj@3 | 34.5 | 49.1 | 36.0 | 0.0 |
| gpt-4o | 33.9 | 47.2 | 34.9 | 3.8 |
| gpt-4o-mini maj@3 | 29.1 | 45.3 | 26.7 | 3.8 |
| gpt-4o-mini | 25.5 | 45.3 | 20.9 | 0.0 |

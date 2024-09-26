# Gaia Agent
The Gaia Agent is an agent that solves the tasks from the [GAIA Benchmark](https://huggingface.co/spaces/gaia-benchmark/leaderboard).
The agent can use OpenAI or big LLAMA models to solve these tasks.

<img width="1232" alt="image" src="https://github.com/user-attachments/assets/f02d9dff-0cfb-4ad6-b061-9d7568d75baa">


## Structure
The agent is built on top of the GuidedAgent class, which implements the following workflow:
- Expose the set of all available actions and thoughts to the model in each prompt
- Render the whole tape into the prompt, trimming only in case when the tape does not fit into the context window
- Based on the end of the current tape, select the short textual guidance prompt that briefly instructs the LLM what to do next
- Append the hints about formatting to the end of the prompt.

This way, the agent is free to choose which thoughts and actions to use to satisfy the current guidance recommendations without the strict constraints of the state machine.
Additionally, the Gaia agent implements the initial planning step, which produces a «plan» in the form of a sequence of free-form descriptions of the actions that should be taken to solve the task.

## Quickstart
- `python -m examples.gaia_agent.scripts.evaluate` - script to run evaluation on the validation set.
- `python -m examples.gaia_agent.scripts.tape_browser` - Gradio UI for exploring the tapes and metrics produced during evaluation. 

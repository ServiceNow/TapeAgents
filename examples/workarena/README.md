
# WorkArena Agents

This example introduces two agents that solve the tasks from the [Workarena Benchmark](https://github.com/ServiceNow/WorkArena).
The agents could use OpenAI or big LLAMA models to solve these tasks.

<img width="1233" alt="image" src="https://github.com/user-attachments/assets/9e4bc7e5-5547-41cb-aa5f-374c72669da2">

## Structure

Both agents are built using the [MonoNode](../../tapeagents/nodes.py), which implement the following workflow:

- Expose the set of all available actions and thoughts to the model in each prompt
- Render the whole tape into the prompt, trimming only in case when the tape does not fit into the context window
- Based on the end of the current tape, select the short textual guidance prompt that briefly instructs the LLM what to do next
- Append the hints about formatting to the end of the prompt.

The agent is free to choose which thoughts and actions to use to satisfy the current guidance recommendations without the additional constraints of specific node or subagent.

The first agent attempts to replicate the structure of the [original WorkArena Agent](https://github.com/ServiceNow/AgentLab/tree/main/src/agentlab/agents), using the same prompts but slightly different input/output format, as we're sticking to json for the step parsing.
The second agent attempts to solve the benchmark more similarly to the [Gaia Agent example](../gaia_agent), using minimal guidance but attempting to do reflection after performing each new action and observing the modified state of the webpage.

## Quickstart

To run the agents, you need to create the ServiceNow instance first! Please follow the setup instructions in the [Workarena repo](https://github.com/ServiceNow/WorkArena?tab=readme-ov-file#getting-started).  

When the setup is done and all environment variables are set, you can run the agents using the following commands:

- `uv run -m examples.workarena.evaluate` - script to run agent evaluation on the validation set.
- `uv run -m examples.workarena.tape_browser` - Gradio UI for exploring the tapes with screenshots, videos and metrics produced during evaluation.

Agents are configured using hydra configs from the `conf/` directory. You can change the agent configuration by modifying the `conf/workarena_openai.yaml` file.

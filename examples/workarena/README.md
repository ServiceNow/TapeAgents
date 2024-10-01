
# WorkArena Agents
This example introduces two agents that solve the tasks from the [Workarena Benchmark](https://github.com/ServiceNow/WorkArena).
The agents could use OpenAI or big LLAMA models to solve these tasks.

<img width="1233" alt="image" src="https://github.com/user-attachments/assets/9e4bc7e5-5547-41cb-aa5f-374c72669da2">

## Structure
Both agents are built on top of the [GuidedAgent](../../tapeagents/guided_agent.py) class, which implements the following workflow:
- Expose the set of all available actions and thoughts to the model in each prompt
- Render the whole tape into the prompt, trimming only in case when the tape does not fit into the context window
- Based on the end of the current tape, select the short textual guidance prompt that briefly instructs the LLM what to do next
- Append the hints about formatting to the end of the prompt.

The agent is free to choose which thoughts and actions to use to satisfy the current guidance recommendations without the additional constraints of specific node or subagent.

The first agent attempts to replicate the structure of the [original WorkArena Agent](https://github.com/ServiceNow/AgentLab/tree/main/src/agentlab/agents), using the same prompts but slightly different input/output format, as we're sticking to json for the step parsing.
The second agent attempts to solve the benchmark more similarly to the [Gaia Agent example](../gaia_agent), using minimal guidance but attempting to do reflection after performing each new action and observing the modified state of the webpage.

## Quickstart
- `python -m examples.workarena.evaluate` - script to run evaluation on the validation set.
- `python -m examples.workarena.tape_browser` - Gradio UI for exploring the tapes with screenshots, videos and metrics produced during evaluation.

# TapeAgents Examples

Full list of examples of how to use the TapeAgents framework for building, debugging and improving agents. Each example is a self-contained Python script that demonstrates how to use the framework to build an agent for a specific task.

## Run

```zsh
# uv run -m examples.<MODULE> <ARGS>
uv run -m examples.llama_agent
```

## Examples

`module` and `folder/`, ordered by increasing complexity:

- [llama_agent](llama_agent.py) - simplest agent that uses LLaMA model to answer to user in a style of Shakespeare.
- [llama_user](llama_user.py) - conversation between the LLaMA agent and the agent that emulates the user behavior.
- [continue_tapes](continue_tapes.py) - agent that continues the tape on behalf of different user.
- [batch_main_loop](batch_main_loop.py) - batch processing of the tapes.
- [batch_add_observations](batch_add_observations.py) - batch processing when emulating multiple users.
- [chat](chat.py) - demo of two agents chatting with each other.
- [openai_function_calling](openai_function_calling.py) - agent that uses OpenAI API function calling to report weather in a city.
- [openai_function_calling_demo](openai_function_calling_demo.py) - interactice Gradio demo of the previous agent.
- [tools_demo](tools_demo.py) - demo of the previous agent with external web search tool.
- [agent](agent.py) - examples of using nodes to control the agent.
- [code_chat](code_chat.py) - simple agent that can solve tasks using python code.
- [annotator](annotator.py) - example of the agent that annotates the existing tape with some score or label.
- [annotator_demo](annotator_demo.py) - interactive Gradio demo of the previous agent.
- [multi_chat](multi_chat.py) - multi-agent setup where team of the agents collaborates to answer to user. Includes group manager agent, software developer agent and code executor agent.
- [data_science](data_science/) - data-science oriented multi-agent setup that solve a single data processing task using python.
- [workarena/](workarena/) - custom agent that solves WorkArena benchmark using BrowserGym environment.
- [gaia_agent/](gaia_agent/) - custom agent that solves Gaia benchmark using planning and a set of tools with web search, documents and media parsers, code execution.
- [delegate](delegate.py) - multi-agent setup where agent contains multiple sub-agents and delegates the tasks to them.
- [delegate_stack](delegate_stack.py) - more complex multi-agent setup where agent uses stack of the tape views to track data for each sub-agent.
- [studio](studio.py) - interactive Gradio demo of agent that could be edited in runtime.
- [tape_improver/](tape_improver/) - the agent that revisit and improves the tapes produced by another agent.
- [gsm8k_tuning/](gsm8k_tuning/) - custom agent that solves GSM-8k benchmark, collect tapes and finetune smaller LLaMA model on them.

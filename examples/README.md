
Full list of examples of how to use the TapeAgents framework for building, debugging and improving agents. Each example is a self-contained Python script that demonstrates how to use the framework to build an agent for a specific task.

## Examples
In the order of increasing complexity:

- [llama_agent.py](examples/llama_agent.py) - simplest agent that uses LLaMA model to answer to user in a style of Shakespeare.
- [llama_user.py](examples/llama_user.py) - conversation between the LLaMA agent and the agent that emulates the user behavior.
- [continue_tapes.py](examples/continue_tapes.py) - agent that continues the tape on behalf of different user.
- [batch_main_loop.py](examples/batch_main_loop.py) - batch processing of the tapes.
- [batch_add_observations.py](examples/batch_add_observations.py) - batch processing when emulating multiple users.
- [chat.py](examples/chat.py) - demo of two agents chatting with each other.
- [openai_function_calling.py](examples/openai_function_calling.py) - agent that uses OpenAI API function calling to report weather in a city.
- [openai_function_calling_demo.py](openai_function_calling_demo.py) - interactice Gradio demo of the previous agent.
- [tools_demo.py](examples/tools_demo.py) - demo of the previous agent with external web search tool.
- [flow.py](examples/flow.py) - examples of using nodes for control flow in the agent.
- [code_chat.py](examples/code_chat.py) - simple agent that can solve tasks using python code.
- [annotator.py](annotator.py) - example of the agent that annotates the existing tape with some score or label.
- [annotator_demo.py](examples/annotator_demo.py) - interactive Gradio demo of the previous agent.
- [multi_chat.py](examples/multi_chat.py) - multi-agent setup where team of the agents collaborates to answer to user. Includes group manager agent, software developer agent and code executor agent.
- [data_science.py](examples/data_science.py) - data-science oriented multi-agent setup that solve a single data processing task using python.
- [workarena](examples/workarena) - custom agent that solves WorkArena benchmark using BrowserGym environment.
- [gaia_agent](examples/gaia_agent) - custom agent that solves Gaia benchmark using planning and a set of tools with web search, documents and media parsers, code execution.
- [delegate.py](examples/delegate.py) - multi-agent setup where agent contains multiple sub-agents and delegates the tasks to them.
- [delegate_stack.py](examples/delegate_stack.py) - more complex multi-agent setup where agent uses stack of the tape views to track data for each sub-agent.
- [studio.py](examples/studio.py) - interactive Gradio demo of agent that could be edited in runtime.
- [tape_improver.py](examples/tape_improver.py) - the agent that revisit and improves the tapes produced by another agent.
- [gsm8k_tuning](examples/gsm8k_tuning) - custom agent that solves GSM-8k benchmark, collect tapes and finetune smaller LLaMA model on them.


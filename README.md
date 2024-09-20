<div align="center">

# TapeAgents

![Supported Python Versions](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)

</div>

*TapeAgents* is an experimental framework to build, debug, serve and optimize AI agents. The key concept of the framework is *Tape*: a complete semantic-level log of the agent's session. All Agent-Environment interactions are mediated by the orchestrator and must go through the tape

![image](/assets/overview.png)

Key features:
- Build your agent as a low-level state machine, as a high-level multi-agent team configuration, or as a mono-agent guided by multiple prompts
- Debug your agent with TapeAgent studio or TapeBrowser apps
- Serve your agent with response streaming
- Optimize your agent's configuration using successful tapes; finetune the LLM using revised tapes.

The Tape-centric design of TapeAgents will help you at all stages of your project:
- Build with ultimate flexibility of having access to tape for making prompts and generating next steps
- Change your prompts or team structure and resume  the debug session as long as the new agent can continue from the older tape
- Fully control the Agent's tape and the Agent's acting when you use a TapeAgent in an app
- Optimize tapes and agents using the carefully crafted metadata structure that links together tapes, steps, llm calls and agent configurations

# Get Started

We highly recommend starting with the [introductory Jupyter notebook](/intro.ipynb). The notebook will introduce you to all the core concepts of framework. 

# Installation

1. Clone the repository:
```
git clone https://github.com/ServiceNow/TapeAgents.git
```
2. Create and activate a new conda environment:
```
conda create -y -n tapeagents python=3.10
conda activate tapeagents
```
3. Install the package in editable mode from the current directory:
```
pip install -e .
```

# Examples

In the coming days we will present to you the following examples:

- How to build teams of TapeAgents with [AutoGen](https://github.com/microsoft/autogen)-style low-code programming paradigm
- How to finetune a TapeAgent with a small LLM to be better at math problem solving
- An agent that searches the web and uses code interpreter to answer precise questions. We built this agent to solves tasks from the [GAIA challenge](https://huggingface.co/spaces/gaia-benchmark/leaderboard)


The [examples/](examples/) directory contains examples of how to use the TapeAgents framework for building, debugging and improving agents. Each example is a self-contained Python script that demonstrates how to use the framework to build an agent for a specific task.

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
- [develop.py](examples/develop.py) - interactive Gradio demo of agent that could be edited in runtime.
- [tape_improver.py](examples/tape_improver.py) - the agent that revisit and improves the tapes from another agent.
- [gsm8k_tuning](examples/gsm8k_tuning) - custom agent that solves GSM-8k benchmark, collect tapes and finetune smaller LLaMA model on them.



# Learn more 

See our [Now AI paper](https://servicenow.sharepoint.com/sites/snrcat/Shared%20Documents/Forms/AllItems.aspx?id=%2Fsites%2Fsnrcat%2FShared%20Documents%2FTapeAgents%2FTapeAgents%5F2024nowai%2Epdf&parent=%2Fsites%2Fsnrcat%2FShared%20Documents%2FTapeAgents&p=true&ga=1) on TapeAgents.

# Contact

Please use the group email of the Conversational AssistanT (CAT) Program at ServiceNow

snr-cat@servicenow.com

# Acknowledgements

We acknowledge the inspiration we took from prior frameworks, in particular [LangGraph](https://github.com/langchain-ai/langgraph), [AutoGen](https://github.com/microsoft/autogen), [Agents](https://github.com/aiwaves-cn/agents) and [DSPY](https://github.com/stanfordnlp/dspy).



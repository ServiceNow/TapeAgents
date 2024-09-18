TapeAgents is an experimental framework to build, debug, serve and optimize AI agents. The key concept of the framework is Tape: a complete semantic-level log of the agent's session. All Agent-Environment interactions are mediated by the orchestrator and must go through the tape.

![image](/assets/overview.jpg)

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

- How to build teams of TapeAgents with AutoGen-style low-code programming paradigm
- How to finetune a TapeAgent with a small LLM to be better at math problem solving
- An agent that searches the web and uses code interpreter to answer precise questions. We built this agent to solves tasks from the [GAIA challenge](https://huggingface.co/spaces/gaia-benchmark/leaderboard)

# Learn more 

TODO: paper link

# Contact

Please use the group email of the Conversational AssistanT (CAT) Program at ServiceNow

snr-cat@servicenow.com

# Acknowledgements

We acknowledge the inspiration from prior frameworks, in particular LangGraph, AutoGen, Agents and DSPY.



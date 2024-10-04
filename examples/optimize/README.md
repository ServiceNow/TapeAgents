# Agent optimization in TapeAgents

This example demostrates how one can optimize the agent's prompt templates in TapeAgents.

There are two common ways to optimize prompts while keeping the overall structure of the agent the same:
- add demonstrations to a prompt
- change the instruction part of the prompt
In TapeAgents we have a structured prompt template called [LLMFunctionTemplate](tapeagents/llm_function.py) that enables both of these prompt change approaches. If you are familiar with DSPy, you will recognize in this DSPy's signature `Signature` (pun intended). The equivalent of DSPy's modules are `LLMFunctionNode` nodes that apply the respective function template to the tape in order to make the prompt and to generate the next steps.

See our [agent optimization](examples/optimize) example we show how one can build a 2-hop retrieval-augmented generation agent and optimize its query generation prompts using weak supervision. This example is a reimplementation of DSPy intro in TapeAgents.

# How to run optimization example

- rag, bare
- rag, load
- agentic_rag, bare
- agentic_rag, optimize

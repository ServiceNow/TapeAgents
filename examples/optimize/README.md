# Agent optimization in TapeAgents

This example demostrates how one can optimize the agent's prompt templates in TapeAgents.

There are two common ways to optimize prompts while keeping the overall structure of the agent the same:
- add demonstrations to a prompt
- change the instruction part of the prompt
In TapeAgents we have a structured prompt template called [LLMFunctionTemplate](tapeagents/llm_function.py) that enables both of these prompt change approaches. If you are familiar with DSPy, you will recognize in this DSPy's signature `Signature` (pun intended). The equivalent of DSPy's modules are `LLMFunctionNode` nodes that apply the respective function template to the tape in order to make the prompt and to generate the next steps.

See our [agent optimization](examples/optimize) example we show how one can build a 2-hop Retrieval-Augmented Generation agent (a.k.a. agentic RAG) and optimize its query generation prompts using weak supervision. This example is a reimplementation of DSPy intro in TapeAgents. It uses questions from the HotPotQA dataset and the generously provided by DSPy Wikipedia paragraph retrieval service.

# How to run the example

## Setup

First, install extra depedencies:

```bash
pip install -r examples/optimize/requirements.txt
```

## Explore the setting

Go better understand the setup, you can launch a pre-optimized agent in TapeAgents Studio and run it by pressing `Run Loop` button.

```bash
python examples/optimize/optimize.py agent=agentic_rag target=studio load_demos=true  
```

Check out the prompts: they contain support demonstrations of how to use the search engine for complex queries, like this one:

> Context:
N/A
Question: Which of these publications was most recently published, Who Put the Bomp or Self?
Reasoning: Let's think step by step in order to produce the query. We know that publication dates are typically included in metadata for articles or books. By searching for the publication date of each article, we can determine which one was most recently published.
Query: "publication date of Who Put the Bomp" OR "publication date of Self"

That is what we will be learning below.

## Optimize and benchmark different agents

Let's benchmark a basic RAG agent. In the basic RAG the user's question is used as the query.

```bash
$ python -m examples.optimize.optimize agent=rag target=evaluate 
Retrieval accuracy: 0.26
Answer accuracy: 0.54
```

The retrieval accuracy is not that high. Let's try 2-hop Agentic RAG. In our Agentic RAG example the agent makes two retrieval queries, and the second query is based on the paragraphs that were trieved for the first one.

```bash
$ python -m examples.optimize.optimize agent=agentic_rag target=evaluate 
Retrieval accuracy: 0.50
Answer accuracy: 0.62
```

The retrieval accuracy is higher, but we can do better. Let's optimize the agent's prompts using weak supervision.

```bash
$ python -m examples.optimize.optimize agent=agentic_rag optimize.do=true target=evaluate
Retrieval accuracy: 0.56
Answer accuracy: 0.52
```

And this way we get a higher retrieval accuracy, though answer accuracy went down.

Note:
- we found the quantitative results of this experiment to be very unstable due to the LLM non-determinism and the small training and dev set sizes. In our future work we will add validation of the selected examples and evaluate on a larget dev set.
- by default the LLM cache is on, so if you rerun an experiment, you will get the exact same results. You can run another experiment by changing passing `exp_name=<another_name>` to Hydra.

## Explore resulting tapes

Change `target` to `browse` to launch the TapeBrowser app.

```bash
$ python examples/optimize/optimize.py agent=agentic_rag optimize.do=true target=browse
```

You can now explore the agent tapes on the dev set, as well as the "good" and the "bad" training tapes. The good tapes that are the ones we used to mine demonstrations for the function templates. The bad tapes are the ones that we filtered out by various criteria (see `result` field in metadata in the tape browser for the reason for filtering). 


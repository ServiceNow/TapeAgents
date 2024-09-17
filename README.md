## Updated on August 22, 2024

The multi-agent examples now use Podman for container execution. You must install it on your machine (see [instructions](https://podman.io/getting-started/installation.html)).
In some cases you will have to set `DOCKER_HOST` environment variable to make Podman accesible for Tapeagents code, 
e.g. `DOCKER_HOST=http+unix:///var/run/docker.sock`. See the output of `podman machine start` for the path to the socket.

To try a new multi-agent data science example, run 

```
python -m tapeagents.examples.data_science develop 
```

See `outputs/...` folder for the code files and images that the agent generated.

To try the tape enhancer example, run these two commands in separate terminals:

```
python -m tapeagents.examples.tape_enhancer develop agent
python -m tapeagents.examples.tape_enhancer develop improver
```

You can run the transformation in the first app, and then see the tape of the CodeEnhancer agent in the second app.

## Updated on August 16, 2024

To try a multi-agent TapeAgent organization, use this demo:

```
python -m tapeagents.examples.multi_chat develop  
```

Note: you may have to install `arxiv` package in your environment.
Note: sometimes the chat manager gets confused and asks a wrong agent to go next. Try again!

# TapeAgents
TapeAgents is a framework that helps AI Agent Builders improve their agents with methods beyond manual prompting, such as learning in simulation.

<img align="right" width="500" alt="image" src="https://github.com/user-attachments/assets/eff8d259-4d39-4d78-9094-0ca17933e435">

- [TapeAgents](#tapeagents)
  - [Requirements](#requirements)
  - [Installation](#installation)
  - [Quickstart](#quickstart)
  - [Examples](#examples)
    - [Examples to start with](#examples-to-start-with)
    - [Other Examples](#other-examples)
    - [Gaia Agent](#gaia-agent)
  - [Concepts](#concepts)
  - [Workflow](#workflow)
  - [Base Classes](#base-classes)
  - [Observability](#observability)
  - [Caching](#caching)

## Requirements
All framework requirements are listed in `requirements.txt`. You don't need to install them manually, they will be installed during the next step.

Some examples come with their own `requirements.txt` file, use the specific example README in such cases.

## Installation
1. clone the repository:
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

Now you can import required modules from the tapeagents.  
Wheel builds are planned in the near future.

## Quickstart
Folder `examples/` contains common useful examples of agents. You can run, explore and make a copy to use as a starting boilerplate. 

Many examples require a LLAMA model. You can either run it locally using VLLM or use a public API. By default all examples use `https://api.together.xyz` service. You can get a free API key at [together.ai](together.ai) and pass it to TapeAgents as `TAPEAGENTS_LLM_TOKEN` environment variable. For examples using OpenAI you will need to `OPENAI_API_KEY` and `OPENAI_ORGANIZATION` environment variables.

## Examples

### Examples to start with
- `python -m tapeagents.examples.llama_agent`

  LLAMAChatBot is a simple agent that utilizes the LLAMA language model to generate responses in a conversational dialog. The script initializes the LLAMAChatBot with an LLAMA instance, renders prompts and predicted steps, generates events based on user input, creates traces, and performs various checks and operations on the generated dialog.

- `python -m tapeagents.examples.llama_user`
  
  LLAMAUserModel is an agent that mimics user behavior by generating responses to the agent's steps.

### Other Examples
- `python -m tapeagents.examples.batch_main_loop`

  Script that demonstrates how to continue multiple tapes using the same agent in batch mode.
- `python -m tapeagents.examples.annotator`

  Script that demonstrates how to use the judge agent to annotate the steps in the existing tape.
- `python -m tapeagents.examples.annotator_demo`

  Gradio UI for interactive annotator demo.
- `python -m tapeagents.examples.langchain_tools_demo`

  Gradio UI demo for the agent that can use the tools from the Langchain Tools.
- `python -m tapeagents.examples.openai_function_calling`

  Script that demonstrates the agent that uses OpenAI API to call the functions.
- `python -m tapeagents.examples.openai_function_calling_demo`

  Gradio UI demo for the tool calling agent.

### Gaia Agent
The Gaia Agent located in `tapeagents/examples/gaia_agent/` is a more complex agent that can use both OpenAI models and LLAMA models to solve the tasks from the [GAIA Benchmark](https://huggingface.co/spaces/gaia-benchmark/leaderboard).
- `python -m tapeagents.examples.gaia_agent.gaia`

  Script to run evaluation on the validation set.
- `python -m tapeagents.examples.gaia_agent.tape_browser`

  Gradio UI to explore the tapes and metrics produced during evaluation. 
  Note: Gaia Agent requires additional dependencies for its browser and file converter tools, see the [README](/tapeagents/tapeagents/examples/gaia_agent/README.md) in the `examples/gaia_agent` folder for more details.



## Concepts

- **Step**: isolated element of the conversation history. Could be agent thought, agent action, environment reaction, or any other step you define in your agent
- **Tape**: an ordered list of steps with metadata attached. Represents a history of some run of the agent and environment. Steps are added in the order of their creation. Tapes can be shared
between different but related agents. The tape represents both the input and the output of the agent system.
- **Agent**: autonomous object, usually backed by llm, that can parse incoming tape and produce new steps using its `.run(tape)` method
- **Environment**: an object representing interaction with the external world. Accept the tape as input and react to the action steps at the end of the tape. Usually produces `Observation` steps, but that could be changed. In some cases, we simulate the environment with a special type of agent called `ObservationMaker`

## Workflow
Agent usually runs in the loop with the specific environment and some kind of starter tape, that could be empty or contain some initial input data. Main interaction performed in the 3 nested loops:  

<img width="746" alt="image" src="https://github.com/user-attachments/assets/e04233bd-7269-403f-8470-7b8bf8cc5022">

The agent and the environment work in a common main loop. They interact by appending new steps to the tape.

A single run of an agent can perform multiple iterations inside. Each iteration usually makes a separate call to the LLM. A single call can produce any amount of steps. The agent performs such iterations until the `stop_step` has been produced or the maximum number of iterations reached. During the run, the agent yields `AgentEvent`, which contains the newly produced step. At the end of the run, the agent yields the event with the `final_tape`.

When the agent run is finished, the main loop runs the environment `react()` method that can produce observation steps and add them to the tape. The environment also yields events when producing new steps. The main loop can be stopped by the agent or by the environment. The main loop can be run in batch mode when the agent runs multiple tapes in parallel.


## Base Classes
Module `tapeagents.core` contains the following base classes:
- `Tape`: base class for all tapes
- `Step`: base class for all steps in `Tape`
- `Prompt`: base class for all LLM prompts
- `Trace`: trace represents the pair of the prompt and completion that can be used to fine-tune the model
- `Agent`: base class for all agents
- `Environment`: base class for all environments
- `Episode`: Auxiliary data structure for tape with annotations attached

LLM Module `tapeagents.llms` contains the base `LLM` class and two basic implementations:
- `LLAMA`: the class that wraps the TGI or VLLM inference endpoint of the LLAMA model. We need it because API endpoints with broken SSL endpoints cannot be used with LiteLLM. It should be deprecated eventually
- `LiteLLM`: the class that uses the LiteLLM library to warp almost any external LLM endpoint that has OpenAI-compatible API.


## Observability
TapeAgents provides a set of tools to observe the agent's behavior, generated tapes, and LLM calls.
- **Prompts and completions logging**: we store all prompts and completions in the SQLite database. Each prompt is paired with the completion and can be associated with the tape by ID. This allows us to track the history of the agent's behavior and the generated tapes.
- **Gradio UIs for tapes**: we provide a set of Gradio UIs to explore the tapes and metrics produced during the agent evaluation. See the corresponding examples for more details.


## Caching
- **LLM Cache**: we cache the LLM calls to avoid the same calls for the same prompts. Behavior is controlled by the flag `use_cache` passed to the LLM constructor. The cache is stored in the jsonl file so it persists between the runs. The cache is not shared between different LLM instances and different configurations.
- **Search Cache**: some agent in the examples use web search APIs. We cache the search results to avoid making the same calls for the same queries.

# Gaia Agent

The Gaia Agent is designed to answer knowledge-grounded questions using web search, calculations, and reasoning. The agent utilizes LLM models through APIs to accomplish these tasks. It is equipped with the following tools:
- [BrowserGym](https://github.com/ServiceNow/BrowserGym) as the main web browser tool
- Web search tool
- Code executor
- Media reader

We demonstrate how it solves tasks from the [GAIA Benchmark](https://huggingface.co/spaces/gaia-benchmark/leaderboard).

## Structure

The agent is defined using this [config](../../conf/gaia_agent.yaml), which implements the following workflow:

- Expose the set of all available actions and thoughts to the model in each prompt
- Render the whole tape into the prompt, trimming only when the tape does not fit into the context window
- Append a short textual guidance prompt that briefly instructs the LLM on what to do next
- Append hints about formatting to the end of the prompt

The agent is free to choose which thoughts and actions to use to satisfy the current guidance recommendations without additional constraints of a specific node or subagent.

Additionally, the Gaia Agent implements an initial planning step, which produces a "plan" in the form of a sequence of free-form descriptions of the actions that should be taken to solve the task.

## Results

Results are also available on the [Hugging Face Leaderboard](https://huggingface.co/spaces/gaia-benchmark/leaderboard) under the names "TapeAgent ...".

| Model | Validation Accuracy | Test Accuracy |
| --- | --- | --- |
| Sonnet 3.7 maj@3 | 55.8 | |
| Sonnet 3.7 | 53.9 | |
| GPT-4o | 37.0 | 33.2 |
| GPT-4o mini maj@3 | 32.3 | |
| GPT-4o mini | 27.3 | 21.9 |

## Tape Example 
<img width="867" alt="gaia_perfect_demo_tape" src="https://github.com/user-attachments/assets/a81c22d8-9cf5-42c4-a390-933108753966">

## Quickstart

Perform all the following steps from the top folder of the repo.
First, install the dependencies for file converters:

```bash
pip install 'tapeagents[converters]'
```

Then, ensure you have `FFmpeg` version 7.1.x or newer installed (more details [here](https://github.com/kkroening/ffmpeg-python?tab=readme-ov-file#installing-ffmpeg)).

If you want to convert PDF files to images to preserve tables and complex formatting, please install the prerequisites of the pdf2image library as described [in their documentation](https://pypi.org/project/pdf2image/).

Then you can run the agent using the following commands:

- `uv run -m examples.gaia_agent.scripts.studio` - Interactive GUI that allows you to set the task for the agent and observe how it solves it step by step.
- `uv run -m examples.gaia_agent.scripts.evaluate` - Script to run evaluation on the GAIA validation set.
- `uv run -m examples.gaia_agent.scripts.tape_browser` - Gradio UI for exploring the tapes and metrics produced during evaluation.

If you see the error `remote: Access to dataset gaia-benchmark/GAIA is restricted. You must have access to it and be authenticated to access it. Please log in.`, you need to log in to your Hugging Face account first:

```bash
huggingface-cli login
```

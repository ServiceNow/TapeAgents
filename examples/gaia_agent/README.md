# Gaia Agent

The Gaia Agent is an agent that can answer knowledge-grounded questions using web search, calculations and reasoning. The agent can use OpenAI or big LLAMA models to solve these tasks. We demonstrate how it solves the tasks from the [GAIA Benchmark](https://huggingface.co/spaces/gaia-benchmark/leaderboard).

Tape example:  
<img width="867" alt="gaia_perfect_demo_tape" src="https://github.com/user-attachments/assets/a81c22d8-9cf5-42c4-a390-933108753966">

## Structure

The agent is built using the [MonoNode](../../tapeagents/nodes.py), which implement the following workflow:

- Expose the set of all available actions and thoughts to the model in each prompt
- Render the whole tape into the prompt, trimming only in case when the tape does not fit into the context window
- Append a short textual guidance prompt that briefly instructs the LLM what to do next
- Append the hints about formatting to the end of the prompt.

The agent is free to choose which thoughts and actions to use to satisfy the current guidance recommendations without the additional constraints of specific node or subagent.

Additionally, the Gaia agent implements the initial planning step, which produces a «plan» in the form of a sequence of free-form descriptions of the actions that should be taken to solve the task.

## Quickstart

Perform all the following steps from the top folder of the repo.
First, you need to install the dependencies for file converters:

```bash
pip install 'tapeagents[converters]'
```

Then, make sure you have `FFmpeg` version 7.1.x or newer installed (more details [here](https://github.com/kkroening/ffmpeg-python?tab=readme-ov-file#installing-ffmpeg)).

If you want to use conversion of PDF files to images to preserve tables and complex formatting, please install prerequisites of the pdf2image library as described [in their documentation](https://pypi.org/project/pdf2image/).

Then you can run the agent using the following commands:

- `uv run -m examples.gaia_agent.scripts.studio` - Interactive GUI that allows you to set the task for the agent and observe how it solves it step by step.
- `uv run -m examples.gaia_agent.scripts.evaluate` - script to run evaluation on the GAIA validation set.
- `uv run -m examples.gaia_agent.scripts.tape_browser` - Gradio UI for exploring the tapes and metrics produced during evaluation.

If you see the error `remote: Access to dataset gaia-benchmark/GAIA is restricted. You must have access to it and be authenticated to access it. Please log in.`, you need to log in to the Hugging Face account first:

```bash
huggingface-cli login
```

You can adjust config in [conf/gaia_openai.yaml](../../conf/gaia_openai.yaml).

## Results

Results are also available on the [Hugging Face Leaderboard](https://huggingface.co/spaces/gaia-benchmark/leaderboard) under the name TapeAgent v0.2.

| Model | Validation Accuracy | Test Accuracy |
| --- | --- |  --- |
| gpt-4o | 37.0 | 33.2 |
| gpt-4o-mini maj@3 | 32.3 |  |
| gpt-4o-mini | 27.3 | 21.9 |

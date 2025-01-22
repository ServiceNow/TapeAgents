# Multi-agent collaboration with code execution for data science

This example show an AutoGen-like multi-agent collaboration. There are 3 AI Agents (`SoftwareEngineer`, `CodeExecutor`, `GroupChatManager`) that collaborates with a human (`UserProxy`).

## Structure

The agentic system is built on top of the [TeamAgent](../../tapeagents/team.py) class, which expose the following:

- `create()` agent that can execute code, think and respond to messages.
- `create_team_manager()` that broadcasts the last message to all subagents, selects one of them to call, call it and responds to the last message if the termination message is not received.
- `create_chat_initiator()` sets the team's initial message and calls the team manager

## Setup

The `CodeExecutor` use `Podman` for container execution. You must install it on your machine (see [instructions](https://podman.io/getting-started/installation.html)).

In some cases you will have to set `DOCKER_HOST` environment variable to make Podman accesible for TapeAgents code, e.g. `DOCKER_HOST=http+unix:///var/run/docker.sock`. See the output of `podman machine start` for the path to the socket.

## Quickstart

```
uv run -m examples.data_science.data_science studio
```

See [outputs](../../outputs/) folder for the code files and images that the CodeExecutor generated.

# Tape Improver

This example show an Agentic Tape Improver that use a pre-generated tape from the [Data Science Agent](../data_science/) and generate better step. The `CodeImprover` agent is composed of 3 other agents: `AgentSelector`, `StepSelector` and `StepRewriter`.

## Setup

The `CodeExecutor` use `Podman` for container execution. You must install it on your machine (see [instructions](https://podman.io/getting-started/installation.html)).

In some cases you will have to set `DOCKER_HOST` environment variable to make Podman accesible for TapeAgents code, e.g. `DOCKER_HOST=http+unix:///var/run/docker.sock`. See the output of `podman machine start` for the path to the socket.

## Execution

1. In a terminal, run the first app that load a pre-saved bad tape

```bash =
uv run -m examples.tape_improver.tape_improver agent
```

2. Under the dropdown `Run a transform`, select `improve_code`. This run the `CodeImprover` multi-agent which update the tape.

3. In a separated terminal, run the second app:

```bash
uv run -m examples.tape_improver.tape_improver improver
```

4. Copy the field `author_tape_id` from the Tape's Metadata of the first app and paste it in the second app under `Load tape by id`. In the second app, you can now see the tape produced by the `CodeImprover` agent to improve the initial tape.

5. In the first app you can click on `Run Loop` to continue the tape from the improved step.

See [outputs](../../outputs/) folder for the code files and images that the CodeExecutor generated.

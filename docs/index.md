# TapeAgents Documentation

**TapeAgents** is a framework that leverages a structured, replayable log (**Tape**) of the agent session to facilitate all stages of the LLM Agent development lifecycle. In TapeAgents, the agent reasons by processing the tape and the LLM output to produce new thoughts, actions, control flow steps and append them to the tape. The environment then reacts to the agent’s actions by likewise appending observation steps to the tape.

## Contents
- [Quick Start](quickstart.md) - Get started with TapeAgents.
- [Reference](reference/) - Detailed documentation of the TapeAgents framework.
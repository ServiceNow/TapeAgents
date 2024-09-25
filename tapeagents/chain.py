from __future__ import annotations

from typing import Any, Generic

from typing_extensions import Self

from tapeagents.agent import Agent, Node
from tapeagents.core import Tape, TapeType
from tapeagents.llms import LLMStream
from tapeagents.view import Call, Respond, TapeViewStack

AgentInputs = tuple[str | int, ...]


class CallChainAgentNode(Node):
    agent_name: str
    inputs: AgentInputs

    def generate_steps(self, _: Any, tape: Tape, llm_stream: LLMStream):
        view = TapeViewStack.compute(tape)
        yield Call(agent_name=self.agent_name)
        for input_ in self.inputs:
            yield view.top.get_output(input_).model_copy()


class RespondIfNotRootNode(Node):
    def generate_steps(self, _: Any, tape: Tape, llm_stream: LLMStream):
        view = TapeViewStack.compute(tape)
        if len(view.stack) > 1:
            yield Respond(copy_output=True)
        return


class Chain(Agent[TapeType], Generic[TapeType]):
    """Calls agents sequentially. Copies thoughts of previous agents for the next agents."""

    @classmethod
    def create(cls, subagents_with_inputs: list[tuple[Agent, AgentInputs]], **kwargs) -> Self:
        flow = []
        subagents = []
        for subagent, inputs in subagents_with_inputs:
            subagents.append(subagent)
            flow.append(CallChainAgentNode(agent_name=subagent.name, inputs=inputs))
        flow.append(RespondIfNotRootNode())
        return super().create(flow=flow, subagents=subagents, **kwargs)

from __future__ import annotations

from typing import Any, Generic

from pydantic import Field
from typing_extensions import Self

from tapeagents.agent import Agent, Node
from tapeagents.core import Tape, TapeType
from tapeagents.llms import LLMStream
from tapeagents.view import Call, Respond, TapeViewStack


class SubagentCall(Node):
    """
    Node that calls a subagent with inputs from the current tape view.
    """

    agent: Agent
    inputs: tuple[str | int, ...] = Field(
        default_factory=tuple,
        description="Names of the subagents which outputs are required for the current subagent to run",
    )

    def model_post_init(self, __context: Any) -> None:
        self.name = f"{self.agent.name}Node"

    def generate_steps(self, _: Any, tape: Tape, llm_stream: LLMStream):
        view = TapeViewStack.compute(tape)
        yield Call(agent_name=self.agent.name)
        for input_ in self.inputs:
            yield view.top.get_output(input_).model_copy(deep=True)


class RespondIfNotRootNode(Node):
    def generate_steps(self, _: Any, tape: Tape, llm_stream: LLMStream):
        view = TapeViewStack.compute(tape)
        if len(view.stack) > 1:
            yield Respond(copy_output=True)
        return


class Chain(Agent[TapeType], Generic[TapeType]):
    """Calls agents sequentially. Copies thoughts of previous agents for the next agents."""

    @classmethod
    def create(cls, nodes: list[Subagent], **kwargs) -> Self:
        subagents = []
        for node in nodes:
            subagents.append(node.agent)
        return super().create(nodes=nodes + [RespondIfNotRootNode()], subagents=subagents, **kwargs)

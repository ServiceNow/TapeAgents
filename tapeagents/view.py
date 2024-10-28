from __future__ import annotations

from collections import defaultdict
from typing import Generic, Literal

from pydantic import BaseModel, Field

from tapeagents.core import AgentStep, Call, Observation, Respond, SetNextNode, StepType, Tape, Thought


class Broadcast(Thought):
    """Broadcase a message to many subagents.

    The current agent remains active.

    """

    content: str
    from_: str
    to: list[str]
    kind: Literal["broadcast"] = "broadcast"


class TapeView(BaseModel, Generic[StepType]):
    """
    Ephemeral view of an agent's part of the tape.

    Presents tape data in the form that is describing for describing the agent's logic.

    """

    agent_name: str
    agent_full_name: str
    steps: list[StepType] = []
    steps_by_kind: dict[str, list[StepType]] = {}
    outputs_by_subagent: dict[str, StepType] = {}
    last_node: str = ""
    last_prompt_id: str = ""
    next_node: str = ""

    def add_step(self, step: StepType):
        self.steps.append(step)
        kind = step.kind  # type: ignore
        if kind not in self.steps_by_kind:
            self.steps_by_kind[kind] = []
        if isinstance(step, SetNextNode):
            self.next_node = step.next_node
        if isinstance(step, AgentStep):
            self.last_node = step.metadata.node
            if step.metadata.prompt_id != self.last_prompt_id:  # start of the new iteration
                self.next_node = ""
                self.last_prompt_id = step.metadata.prompt_id
        self.steps_by_kind[kind].append(step)

    def get_output(self, subagent_name_or_index: int | str) -> StepType:
        if isinstance(subagent_name_or_index, int):
            return list(self.outputs_by_subagent.values())[subagent_name_or_index]
        return self.outputs_by_subagent[subagent_name_or_index]


class TapeViewStack(BaseModel, Generic[StepType]):
    """
    Stack of tape views of the agents in the call chain.

    If Agent A calls Agent B, and then Agent B calls Agent C,
    the stack will looks as follows:
    0: TapeView of Agent A
    1: TapeView of Agent B
    2: TapeView of Agent C
    """

    stack: list[TapeView[StepType]]
    messages_by_agent: dict[str, list[Call | Respond | Broadcast]] = Field(default_factory=lambda: defaultdict(list))

    @property
    def top(self):
        return self.stack[-1]

    def is_step_by_active_agent(self, step: StepType):
        # state machine doesn't know the name of the root agent, so in the comparison here
        # we need cut of the first component
        if not isinstance(step, AgentStep):
            return False
        parts_by = step.metadata.agent.split("/")
        parts_frame_by = self.top.agent_full_name.split("/")
        return parts_by[1:] == parts_frame_by[1:]

    def update(self, step: StepType):
        top = self.stack[-1]
        match step:
            case Call():
                self.put_new_view_on_stack(step)
            case Broadcast():
                self.broadcast(step)
            case Respond():
                self.pop_view_from_stack(step)
            case AgentStep():
                top.add_step(step)
            case Observation():
                top.add_step(step)
            case _:
                raise ValueError(f"Unsupported step type {step}")

    def pop_view_from_stack(self, step: Respond):
        top = self.stack[-1]
        self.stack.pop()
        new_top = self.stack[-1]

        if step.copy_output:
            for top_step in reversed(top.steps):
                # How we choose the output step of the frame.
                # - exclude the input steps that the caller agent added to the tape for the given agent
                #   (note that by this line the former caller agent is the active agent)
                # - exclude Call and Respond steps
                # - exclude Observation steps
                # - among the remaining steps pick the last one
                if not self.is_step_by_active_agent(top_step) and not isinstance(
                    top_step, (Call, Respond, Observation)
                ):
                    new_top.add_step(top_step)
                    new_top.outputs_by_subagent[top.agent_name] = top_step
                    break
                    # TODO: what if the agent was not called by its immediate manager?

        receiver = step.metadata.agent.rsplit("/", 1)[0]
        self.messages_by_agent[step.metadata.agent].append(step)
        self.messages_by_agent[receiver].append(step)
        new_top.add_step(step)

    def broadcast(self, step):
        top = self.stack[-1]
        top.add_step(step)
        for to in step.to:
            receiver = f"{step.metadata.agent}/{to}"
            self.messages_by_agent[receiver].append(step)

    def put_new_view_on_stack(self, step):
        top = self.stack[-1]
        top.add_step(step)
        self.stack.append(
            TapeView(
                agent_name=step.agent_name,
                agent_full_name=top.agent_full_name + "/" + step.agent_name,
            )
        )
        self.stack[-1].add_step(step)
        receiver = f"{step.metadata.agent}/{step.agent_name}"
        self.messages_by_agent[step.metadata.agent].append(step)
        self.messages_by_agent[receiver].append(step)

    @staticmethod
    def compute(tape: Tape) -> TapeViewStack[StepType]:
        # TODO: retrieve view from a prefix of the tape, recompute from the prefix
        stack = TapeViewStack(stack=[TapeView(agent_name="root", agent_full_name="root")])
        for step in tape.steps:
            stack.update(step)
        return stack  # type: ignore

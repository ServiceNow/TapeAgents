from __future__ import annotations

import logging
from typing import Generator

from pydantic import ConfigDict

from tapeagents.agent import DEFAULT, Agent, AgentStep, Node
from tapeagents.autogen_prompts import SELECT_SPEAKER_MESSAGE_AFTER_TEMPLATE, SELECT_SPEAKER_MESSAGE_BEFORE_TEMPLATE
from tapeagents.container_executor import extract_code_blocks
from tapeagents.core import FinalStep, Jump, Pass, Prompt, StepMetadata, Tape
from tapeagents.environment import CodeExecutionResult, ExecuteCode
from tapeagents.llms import LLM, LLMStream
from tapeagents.view import Broadcast, Call, Respond, TapeViewStack

logger = logging.getLogger(__name__)


TeamTape = Tape[None, Call | Respond | Broadcast | FinalStep | Jump | ExecuteCode | CodeExecutionResult | Pass]


class ActiveTeamAgentView:
    def __init__(self, agent: TeamAgent, tape: TeamTape):
        """
        ActiveTeamAgentView contains the ephemeral state computed from the tape. This class extracts the data relevant to
        the given agent and also computes some additional information from it, e.g. whether the agent
        should call the LLM to generate a message or respond with an already available one.
        """
        view = TapeViewStack.compute(tape)
        self.messages = view.messages_by_agent[agent.full_name]
        self.last_non_empty_message = next((m for m in reversed(self.messages) if m.content), None)
        self.node = agent.get_node(view)
        self.steps = view.top.steps
        self.steps_by_kind = view.top.steps_by_kind
        self.exec_result = self.steps[-1] if self.steps and isinstance(self.steps[-1], CodeExecutionResult) else None
        self.should_generate_message = (
            isinstance(self.node, (CallNode, RespondNode))
            and self.messages
            and not self.exec_result
            and "system" in agent.templates
        )
        self.should_stop = (
            agent.max_calls and (agent.max_calls and len(self.steps_by_kind.get("call", [])) >= agent.max_calls)
        ) or (self.messages and ("TERMINATE" in self.messages[-1].content))


class TeamAgent(Agent[TeamTape]):
    """
    Agent designed to work in the team with similar other agents performing different kinds
    """

    max_calls: int | None = None
    init_message: str | None = None

    model_config = ConfigDict(use_enum_values=True)

    def get_node(self, view: TapeViewStack) -> Node:
        return self.flow[view.top.next_node]

    @classmethod
    def create(
        cls,
        name: str,
        system_prompt: str | None = None,
        llm: LLM | None = None,
        execute_code: bool = False,
    ):  # type: ignore
        """
        Create a simple agent that can execute code, think and respond to messages
        """
        return cls(
            name=name,
            templates={"system": system_prompt} if system_prompt else {},
            llms={DEFAULT: llm} if llm else {},
            flow=([ExecuteCodeNode()] if execute_code else []) + [RespondNode()],
        )

    @classmethod
    def create_team_manager(
        cls,
        name: str,
        subagents: list[Agent[TeamTape]],
        llm: LLM,
        max_calls: int = 1,
    ):
        """
        Create a team manager that broadcasts the last message to all subagents, selects one of them to call, call it and
        responds to the last message if the termination message is not received.
        """
        return cls(
            name=name,
            subagents=subagents,
            flow=[
                BroadcastLastMessageNode(),
                SelectAndCallNode(),
                RespondOrRepeatNode(),
            ],
            max_calls=max_calls,
            templates={
                "select_before": SELECT_SPEAKER_MESSAGE_BEFORE_TEMPLATE,
                "select_after": SELECT_SPEAKER_MESSAGE_AFTER_TEMPLATE,
            },
            llms={DEFAULT: llm},
        )

    @classmethod
    def create_chat_initiator(
        cls,
        name: str,
        teammate: Agent[TeamTape],
        init_message: str,
        system_prompt: str = "",
        llm: LLM | None = None,
        max_calls: int = 1,
        execute_code: bool = False,
    ):
        """
        Create an agent that sets the team's initial message and calls the team manager
        """
        return cls(
            name=name,
            templates={
                "system": system_prompt,
            },
            llms={DEFAULT: llm} if llm else {},
            subagents=[teammate],
            flow=([ExecuteCodeNode()] if execute_code else []) + [CallNode(), TerminateOrRepeatNode()],  # type: ignore
            max_calls=max_calls,
            init_message=init_message,
        )


class BroadcastLastMessageNode(Node):
    name: str = "broadcast_last_message"

    def generate_steps(
        self, agent: TeamAgent, tape: TeamTape, llm_stream: LLMStream
    ) -> Generator[AgentStep, None, None]:
        view = ActiveTeamAgentView(agent, tape)
        recipients = agent.get_subagent_names()
        last_step = view.messages[-1]
        from_ = last_step.metadata.agent.split("/")[-1]
        match last_step:
            case Call():
                yield Broadcast(content=last_step.content, from_=from_, to=list(recipients)).by_node(self.name)
            case Respond():
                recipients = [name for name in recipients if name != last_step.metadata.agent.split("/")[-1]]
                yield Broadcast(
                    content=view.messages[-1].content,
                    from_=from_,
                    to=list(recipients),
                ).by_node(self.name)
            case Broadcast(metadata=StepMetadata(node=self.name)):
                pass
            case _:
                assert False


class CallNode(Node):
    name: str = "call"

    def make_prompt(self, agent: TeamAgent, tape: TeamTape) -> Prompt:
        p = Prompt()
        view = ActiveTeamAgentView(agent, tape)
        if view.should_generate_message:
            system = [{"role": "system", "content": agent.templates["system"]}]
            p = Prompt(messages=system + _llm_messages_from_tape(agent, tape))
        return p

    def generate_steps(
        self, agent: TeamAgent, tape: TeamTape, llm_stream: LLMStream
    ) -> Generator[AgentStep, None, None]:
        view = ActiveTeamAgentView(agent, tape)
        # if last node
        (other,) = agent.subagents
        if view.should_generate_message:
            yield Call(agent_name=other.name, content=llm_stream.get_text()).by_node(self.name)
        elif view.exec_result:
            yield Call(agent_name=other.name, content=_exec_result_message(agent, tape)).by_node(self.name)
        else:
            assert agent.init_message and not view.messages
            yield Call(agent_name=other.name, content=agent.init_message).by_node(self.name)


class SelectAndCallNode(Node):
    name: str = "select_and_call"

    def make_prompt(self, agent: TeamAgent, tape: TeamTape) -> Prompt:
        subagents = ", ".join(agent.get_subagent_names())
        select_before = [
            {
                "role": "system",
                "content": agent.templates["select_before"].format(subagents=subagents),
            }
        ]
        select_after = [
            {
                "role": "system",
                "content": agent.templates["select_after"].format(subagents=subagents),
            }
        ]
        return Prompt(messages=select_before + _llm_messages_from_tape(agent, tape) + select_after)

    def generate_steps(
        self, agent: TeamAgent, tape: TeamTape, llm_stream: LLMStream
    ) -> Generator[AgentStep, None, None]:
        callee_name = llm_stream.get_text()
        # check if the callee is an existing subagent
        _ = agent.find_subagent(callee_name)
        yield Call(agent_name=callee_name).by_node(self.name)


class ExecuteCodeNode(Node):
    name: str = "execute_code"

    def generate_steps(self, agent: TeamAgent, tape: Tape, llm_stream: LLMStream) -> Generator[AgentStep, None, None]:
        assert not llm_stream
        view = ActiveTeamAgentView(agent, tape)
        if view.last_non_empty_message is None:
            yield Pass().by_node(self.name)
        elif code := extract_code_blocks(view.last_non_empty_message.content):
            yield ExecuteCode(code=code).by_node(self.name)
        else:
            yield Pass().by_node(self.name)


class RespondNode(Node):
    name: str = "respond"

    def make_prompt(self, agent: TeamAgent, tape: TeamTape) -> Prompt:
        p = Prompt()
        view = ActiveTeamAgentView(agent, tape)
        if view.should_generate_message:
            system = [{"role": "system", "content": agent.templates["system"]}]
            p = Prompt(messages=system + _llm_messages_from_tape(agent, tape))
        return p

    def generate_steps(
        self, agent: TeamAgent, tape: TeamTape, llm_stream: LLMStream
    ) -> Generator[AgentStep, None, None]:
        view = ActiveTeamAgentView(agent, tape)
        if view.should_generate_message:
            yield Respond(content=llm_stream.get_text()).by_node(self.name)
        elif view.exec_result:
            yield Respond(content=_exec_result_message(agent, tape)).by_node(self.name)
        else:
            logger.info(
                f"Agent {agent.full_name} had to respond with an empty message."
                f" You might want to optimize your orchestration logic."
            )
            yield Respond().by_node(self.name)


class TerminateOrRepeatNode(Node):
    name: str = "terminate_or_repeat"

    def generate_steps(
        self, agent: TeamAgent, tape: TeamTape, llm_stream: LLMStream
    ) -> Generator[AgentStep, None, None]:
        assert not llm_stream
        view = ActiveTeamAgentView(agent, tape)
        if view.should_stop:
            yield FinalStep(reason="Termination message received").by_node(self.name)
        else:
            yield Jump(next_node=0).by_node(self.name)


class RespondOrRepeatNode(Node):
    name: str = "respond_or_repeat"

    def generate_steps(
        self, agent: TeamAgent, tape: TeamTape, llm_stream: LLMStream
    ) -> Generator[AgentStep, None, None]:
        view = ActiveTeamAgentView(agent, tape)
        if view.should_stop:
            yield Respond().by_node(self.name)
        else:
            yield Jump(next_node=0).by_node(self.name)


def _exec_result_message(agent: TeamAgent, tape: TeamTape) -> str:
    view = ActiveTeamAgentView(agent, tape)
    exec_result_message = ""
    if view.exec_result:
        if output := view.exec_result.result.output.strip():
            exec_result_message = f"I ran the code and got the following output:\n\n{output}"
        else:
            exec_result_message = f"I ran the code, the exit code was {view.exec_result.result.exit_code}."
    return exec_result_message


def _llm_messages_from_tape(agent: TeamAgent, tape: TeamTape) -> list[dict[str, str]]:
    view = ActiveTeamAgentView(agent, tape)
    llm_messages = []
    for step in view.messages:
        match step:
            # When we make the LLM messages, we use "kind" == "user" for messages
            # originating from other agents, and "kind" == "assistant" for messages by this agent.
            case Call() if step.metadata.agent == agent.full_name:
                # I called someone
                llm_messages.append({"role": "assistant", "content": step.content})
            case Call():
                # someone called me
                # we exclude empty call messages from the prompt
                if not step.content:
                    continue
                llm_messages.append(
                    {
                        "role": "user",
                        "content": step.content,
                        "name": step.metadata.agent.split("/")[-1],
                    }
                )
            case Respond() if step.metadata.agent == agent.full_name:
                # I responded to someone
                llm_messages.append({"role": "assistant", "content": step.content})
            case Respond():
                # someone responded to me
                who_returned = step.metadata.agent.split("/")[-1]
                llm_messages.append({"role": "user", "content": step.content, "name": who_returned})
            case Broadcast():
                llm_messages.append({"role": "user", "content": step.content, "name": step.from_})
    return llm_messages

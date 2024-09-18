from __future__ import annotations

import logging
from enum import Enum
from typing import Literal

from pydantic import ConfigDict

from tapeagents.agent import DEFAULT, Agent
from tapeagents.autogen_prompts import SELECT_SPEAKER_MESSAGE_AFTER_TEMPLATE, SELECT_SPEAKER_MESSAGE_BEFORE_TEMPLATE
from tapeagents.core import Action, FinalStep, Jump, Observation, Prompt, Tape, Pass
from tapeagents.environment import CodeExecutionResult, Environment, ExecuteCode
from tapeagents.llms import LLM, LLMStream
from tapeagents.container_executor import CodeBlock, CommandLineCodeResult, ContainerExecutor, extract_code_blocks
from tapeagents.view import Broadcast, Call, Respond, TapeViewStack

logger = logging.getLogger(__name__)


CollectiveTape = Tape[None, Call | Respond | Broadcast | FinalStep | Jump | ExecuteCode | CodeExecutionResult | Pass]


class Task(str, Enum):
    """
    List of tasks that the agent from the collective can perform
    """

    broadcast_last_message = "broadcast_last_message"
    call = "call"
    select_and_call = "select_and_call"
    execute_code = "execute_code"
    respond = "respond"
    terminate_or_repeat = "terminate_or_repeat"
    respond_or_repeat = "respond_or_repeat"


class ActiveCollectiveAgentView:
    def __init__(self, agent: CollectiveAgent, tape: CollectiveTape):
        """
        CollectiveTapeView contains the ephemeral state computed from the tape. This class extracts the data relevant to
        the given agent and also computes some additional information from it, e.g. whether the agent
        should call the LLM to generate a message or respond with an already available one.

        """
        view = TapeViewStack.compute(tape)
        self.messages = view.messages_by_agent[agent.full_name]
        self.last_non_empty_message = next((m for m in reversed(self.messages) if m.content), None)
        self.task = agent.get_task(view)
        self.steps = view.top.steps
        self.steps_by_role = view.top.steps_by_role
        self.exec_result = self.steps[-1] if self.steps and isinstance(self.steps[-1], CodeExecutionResult) else None
        self.should_generate_message = (
            self.task in {Task.call, Task.respond}
            and self.messages
            and not self.exec_result
            and "system" in agent.templates
        )
        self.should_stop = (
            agent.max_calls and (agent.max_calls and len(self.steps_by_role.get("call", [])) >= agent.max_calls)
        ) or (self.messages and ("TERMINATE" in self.messages[-1].content))


class CollectiveAgent(Agent[CollectiveTape]):
    """
    Agent designed to work in the collective with similar other agents performing different roles
    """

    max_calls: int | None = None
    init_message: str | None = None
    tasks: list[Task]

    model_config = ConfigDict(use_enum_values=True)

    def get_task(self, view: TapeViewStack) -> Task:
        # TODO: use nodes
        return Task(self.tasks[view.top.next_node])

    def make_prompt(self, tape: CollectiveTape) -> Prompt:
        view = ActiveCollectiveAgentView(self, tape)
        llm_messages = []
        for step in view.messages:
            match step:
                # When we make the LLM messages, we use "role" == "user" for messages
                # originating from other agents, and "role" == "assistant" for messages by this agent.
                case Call() if step.by == self.full_name:
                    # I called someone
                    llm_messages.append({"role": "assistant", "content": step.content})
                case Call():
                    # someone called me
                    # we exclude empty call messages from the prompt
                    if not step.content:
                        continue
                    llm_messages.append({"role": "user", "content": step.content, "name": step.by.split("/")[-1]})
                case Respond() if step.by == self.full_name:
                    # I responded to someone
                    llm_messages.append({"role": "assistant", "content": step.content})
                case Respond():
                    # someone responded to me
                    who_returned = step.by.split("/")[-1]
                    llm_messages.append({"role": "user", "content": step.content, "name": who_returned})
                case Broadcast():
                    llm_messages.append({"role": "user", "content": step.content, "name": step.from_})
        match view.task:
            case Task.select_and_call:
                subagents = ", ".join(self.get_subagent_names())
                select_before = [
                    {"role": "system", "content": self.templates["select_before"].format(subagents=subagents)}
                ]
                select_after = [
                    {"role": "system", "content": self.templates["select_after"].format(subagents=subagents)}
                ]
                return Prompt(messages=select_before + llm_messages + select_after)
            case _ if view.should_generate_message:
                system = [{"role": "system", "content": self.templates["system"]}]
                return Prompt(messages=system + llm_messages)
            case _:
                return Prompt()

    @classmethod
    def create(cls, name: str, system_prompt: str | None = None, llm: LLM | None = None, execute_code: bool = False):  # type: ignore
        """
        Create a simple agent that can execute code, think and respond to messages
        """
        return cls(
            name=name,
            templates={"system": system_prompt} if system_prompt else {},
            llms={DEFAULT: llm} if llm else {},
            tasks=([Task.execute_code] if execute_code else []) + [Task.respond],
        )

    @classmethod
    def create_collective_manager(cls, name: str, subagents: list[Agent[CollectiveTape]], llm: LLM, max_calls: int = 1):
        """
        Create a collective manager that broadcasts the last message to all subagents, selects one of them to call, call it and
        responds to the last message if the termination message is not received.
        """
        return cls(
            name=name,
            subagents=subagents,
            tasks=[Task.broadcast_last_message, Task.select_and_call, Task.respond_or_repeat],
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
        collective_manager: Agent[CollectiveTape],
        init_message: str,
        system_prompt: str = "",
        llm: LLM | None = None,
        max_calls: int = 1,
        execute_code: bool = False,
    ):
        """
        Create an agent that sets the collective's initial message and calls the collective manager
        """
        return cls(
            name=name,
            templates={
                "system": system_prompt,
            },
            llms={DEFAULT: llm} if llm else {},
            subagents=[collective_manager],
            tasks=([Task.execute_code] if execute_code else []) + [Task.call, Task.terminate_or_repeat],
            max_calls=max_calls,
            init_message=init_message,
        )

    def generate_steps(self, tape: CollectiveTape, llm_stream: LLMStream):
        view = ActiveCollectiveAgentView(self, tape)

        exec_result_message = ""
        if view.exec_result:
            if output := view.exec_result.result.output.strip():
                exec_result_message = f"I ran the code and got the following output:\n\n{output}"
            else:
                exec_result_message = f"I ran the code, the exit code was {view.exec_result.result.exit_code}."

        def _implementation():
            match view.task:
                case Task.broadcast_last_message:
                    recipients = self.get_subagent_names()
                    last = view.messages[-1]
                    from_ = last.by.split("/")[-1]
                    match last:
                        case Call():
                            yield Broadcast(content=last.content, from_=from_, to=list(recipients))
                        case Respond():
                            recipients = [name for name in recipients if name != last.by.split("/")[-1]]
                            yield Broadcast(content=view.messages[-1].content, from_=from_, to=list(recipients))
                        case Broadcast():
                            pass
                        case _:
                            assert False
                case Task.select_and_call:
                    callee_name = llm_stream.get_text()
                    # check if the callee is an existing subagent
                    _ = self.find_subagent(callee_name)
                    yield Call(agent_name=callee_name)
                case Task.call:
                    # if last task
                    (other,) = self.subagents
                    if view.should_generate_message:
                        yield Call(agent_name=other.name, content=llm_stream.get_text())
                    elif view.exec_result:
                        yield Call(agent_name=other.name, content=exec_result_message)
                    else:
                        assert self.init_message and not view.messages
                        yield Call(agent_name=other.name, content=self.init_message)
                case Task.execute_code:
                    assert not llm_stream
                    if view.last_non_empty_message is None:
                        yield Pass()
                    elif code := extract_code_blocks(view.last_non_empty_message.content):
                        yield ExecuteCode(code=code)
                    else:
                        yield Pass()
                case Task.respond:
                    if view.should_generate_message:
                        yield Respond(content=llm_stream.get_text())
                    elif view.exec_result:
                        yield Respond(content=exec_result_message)
                    else:
                        logger.info(
                            f"Agent {self.full_name} had to respond with an empty message."
                            f" You might want to optimize your orchestration logic."
                        )
                        yield Respond()
                case Task.terminate_or_repeat:
                    assert not llm_stream
                    if view.should_stop:
                        yield FinalStep(reason="Termination message received")
                    else:
                        yield Jump(next_node=0)
                case Task.respond_or_repeat:
                    if view.should_stop:
                        yield Respond()
                    else:
                        yield Jump(next_node=0)
                case _:
                    raise ValueError()

        for step in _implementation():
            step.task = view.task.value
            yield step
        return

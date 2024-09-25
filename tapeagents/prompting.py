from tapeagents.agent import Agent
from tapeagents.core import AgentStep, Call, Jump, Observation, Pass, Respond, Step, Tape
from tapeagents.dialog_tape import SystemStep, ToolResult
from tapeagents.view import TapeView

def step_to_message(step: Step, agent: Agent | None = None) -> dict[str, str]:
    if agent is None and isinstance(step, (Call, Respond)):
        raise ValueError("must known the agent to convert Call or Respond to message")
    
    name = None
    content = step.llm_dict()
    match step:
        case SystemStep():
            role = "system"
        case ToolResult():
            role = "tool"
        case Call() if step.by == agent.full_name: # type: ignore
            role = "assistant"
            content["content"] = f"Call {step.agent_name} with the message '{step.content}'" 
        case Call():
            role = "user"
            name = step.by.split("/")[-1]
        case Respond():
            # use this prompt-making utility only for agents that respond only once,
            # agents that do not have a conversation history
            assert step.by != agent.full_name # type: ignore
            role = "user"
            name = step.by.split("/")[-1]
        case AgentStep():
            role = "assistant"
        case Observation():
            role = "user"
        case _:
            raise ValueError(f"Cannot convert step type: {step} to role")
    llm_message = {"role": role, **content}
    if name:
        llm_message["name"] = name
    return llm_message


def tape_to_messages(tape: Tape, agent: Agent | None = None) -> list[dict]:
    """The default way of representing tape steps as LLM messages."""
    messages = []
    for step in tape.steps:
        if isinstance(step, (Pass, Jump)):
            continue
        llm_message = step_to_message(step, agent)
        messages.append(llm_message)
    return messages


def view_to_messages(view: TapeView, agent: Agent | None = None) -> list[dict]:
    messages = []
    for step in view.steps:
        assert not isinstance(step, (Pass, Jump))
        llm_message = step_to_message(step, agent)
        messages.append(llm_message)
    return messages
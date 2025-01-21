import json
import pathlib
import sys
from typing import Any, Literal

from tapeagents.agent import Agent
from tapeagents.core import (
    Action,
    AgentStep,
    FinalStep,
    Prompt,
    Tape,
    TapeMetadata,
    Thought,
)
from tapeagents.llms import LLM, LiteLLM, LLMStream
from tapeagents.nodes import CallSubagent
from tapeagents.observe import observe_tape
from tapeagents.renderers.pretty import PrettyRenderer
from tapeagents.team import Chain, TeamTape
from tapeagents.view import Call, Respond

from ..data_science.data_science import make_renderers, make_world as data_science_make_world

### Prompts ###

MUST_BE_JSON: str = """Output ONLY the JSON in the requested format. Do not output any text before or after JSON. Do not output triple quotes before or after JSON."""

CONTEXT_TAPE_PREFIX: str = """You are observing a history of a team of AI agents working together. We will refer to this history as 'tape'. The tape
consists of steps taken by different agents. The steps may include reasoning thoughts, messages to each other, as well as
observations coming from the environment. The steps by the agents on the team all have a 'by' attribute with the hierarchical name of the agent.
As the agent are arranged in an hierarchy, the 'by' attribute also includes of the parents of the agents. Here's the tape:

{context_tape}
"""

SELECT_AGENT_SUFFIX: str = """You will select the agent that writes code in this organization.

If there is ONLY ONE agent in the organization that writes code, you will select that agent and output this JSON:

{
    "agent_name": <hierarchical name of the agent>,
    "reason": ""
}

If there is NO agents in the organization that write code, you will output this JSON:

{
    "agent_name": null,
    "reason": "No agent writes code in this organization"
}

If there are MULTIPLE agents in the organization that write code, you will output this JSON:

{
    "agent_name": null,
    "reason": <1-3 sentences explaining what are the multiple agent that you think are writing code>
}

"""

SELECT_STEP_SUFFIX: str = """Based on the feedback {agent_name} is getting for the code they write
select the step in which this agent wrote code that needs improvement. Improvements can include
- adding commands to install missing dependencies
- fixing logical errors

You will output this JSON:

{{
    "reasoning": <spell out your thoughts>
    "step_index": <index of the first step in the tape where the agent writes bad code>
}}

If the agent always wrote great code, you will output "null" in the "index" field.

"""

REWRITE_STEP_SUFFIX: str = """In the previous steps you found that the agent {agent_name} wrote bad code at step {step_index}.

    Your reasoning was as follows: {reasoning}.

    You will rewrite the step in which the agent wrote bad code. Use the feedback that the agent received
    later in the tape after Step {step_index}. If you want the user to take some steps before rerunning the code,
    express these steps as an executable code block.

    You must respect the schema of the step. Just rewrite the field of the step that contains bad code.
    Keep the index field and other metadata field.

    """

### Available Steps ###


class SelectAgent(Thought):
    """
    The step where the improver agent selects the name of the agent from the original tape which steps need to be improved.
    """

    kind: str = "select_agent"
    agent_name: str | None
    reason: str


class SelectStep(Thought):
    """
    The step where the improver agent selects the index of the step from the original tape which needs to be improved.
    """

    kind: str = "select_step"
    reasoning: str
    step_index: int | None


class RewriteStep(Action):
    """
    The step where the improver agent rewrites the step from the original tape with the improved version.
    """

    kind: str = "rewrite_step"
    step_index: int
    # code can be written in either Call or Respond messages
    new_step: Call | Respond


class StepParsingError(Action):
    """
    Step that represents an error in parsing of the imoroved step to inform the improver agent about its fault.
    """

    kind: str = "step_parsing_error"
    error: str


CodeImproverTape = Tape[TeamTape, SelectAgent | SelectStep | RewriteStep | FinalStep | Call | Respond]


def improver_tape_view(tape: Tape) -> str:
    """
    Returns a json representation of the tape for the user to view.
    """
    data = []
    for index, step in enumerate(tape):
        data.append(step.llm_dict())
        data[-1]["index"] = index
        if isinstance(step, AgentStep):
            data[-1]["metadata"] = {"agent": step.metadata.agent}
    return json.dumps(data, indent=2, sort_keys=True)


### Agents ###


class AgentSelector(Agent):
    """
    Agent that selects the agent that wrote code in the given tape.
    Produces the SelectAgent thought containing the name of the agent that writes code.
    """

    def make_prompt(self, tape: CodeImproverTape) -> Prompt:
        assert tape.context is not None
        usr_msg = CONTEXT_TAPE_PREFIX.format(context_tape=improver_tape_view(tape.context))
        usr_msg += SELECT_AGENT_SUFFIX
        usr_msg += MUST_BE_JSON
        return Prompt.from_user_message(usr_msg)

    def generate_steps(self, tape: CodeImproverTape, llm_stream: LLMStream):
        yield SelectAgent.model_validate_json(llm_stream.get_text())
        yield Respond(copy_output=True)


class StepSelector(Agent):
    """
    Agent that selects the step in the tape that needs to be improved.
    Produces the SelectStep thought containing the index of the step that needs improvement.
    """

    def make_prompt(self, tape: CodeImproverTape) -> Prompt:
        assert isinstance(select_step := tape.steps[-1], SelectAgent)
        if select_step.agent_name is None:
            return Prompt()
        assert tape.context is not None
        usr_msg = CONTEXT_TAPE_PREFIX.format(context_tape=improver_tape_view(tape.context))
        usr_msg += SELECT_STEP_SUFFIX.format(agent_name=select_step.agent_name)
        usr_msg += MUST_BE_JSON
        return Prompt.from_user_message(usr_msg)

    def generate_steps(self, tape: Any, llm_stream: LLMStream):
        if not llm_stream:
            yield FinalStep()
        yield SelectStep.model_validate_json(llm_stream.get_text())
        yield Respond(copy_output=True)


class StepRewriter(Agent):
    """
    Agent that rewrites the step in the tape that needs improvement.
    Produces the RewriteStep action containing the new step with the improved code.
    """

    def make_prompt(self, tape: CodeImproverTape) -> Prompt:
        assert isinstance(select_agent := tape.steps[-2], SelectAgent)
        assert isinstance(select_step := tape.steps[-1], SelectStep)
        if select_agent.agent_name is None or select_step.step_index is None:
            return Prompt()
        assert tape.context is not None
        usr_msg = CONTEXT_TAPE_PREFIX.format(context_tape=improver_tape_view(tape.context))
        usr_msg += REWRITE_STEP_SUFFIX.format(
            agent_name=select_agent.agent_name,
            step_index=select_step.step_index,
            reasoning=select_step.reasoning,
        )
        usr_msg += MUST_BE_JSON
        return Prompt.from_user_message(usr_msg)

    def generate_steps(self, tape: Any, llm_stream: LLMStream):
        if not llm_stream:
            yield FinalStep()
        assert isinstance(select_step := tape.steps[-1], SelectStep)
        data = json.loads(llm_stream.get_text())
        assert select_step.step_index is not None
        try:
            new_step = tape.context.steps[select_step.step_index].model_validate(data)
            # deterministic step id to make this example testable
            new_step.metadata.id = "123"
            yield RewriteStep(step_index=select_step.step_index, new_step=new_step)
        except Exception as e:
            yield StepParsingError(error=str(e))


def make_world(llm: LLM | None = None) -> tuple[Agent, Tape, Tape]:
    """
    Creates and initializes the world for the tape improver example.

    Args:
        llm (LLM | None): An optional language model instance. If not provided, a default LiteLLM instance is created.

    Returns:
        tuple[Agent, Tape, Tape]: A tuple containing the code improver agent, the bad tape, and the improver tape to run the agent on.
    """
    res_dir = f"{pathlib.Path(__file__).parent.resolve()}/res"
    with open(f"{res_dir}/bad_tape.json", "r") as f:
        bad_tape = TeamTape.model_validate(json.load(f))
    improver_tape = CodeImproverTape(context=bad_tape, steps=[])

    llm = llm or LiteLLM(
        model_name="gpt-4o",
        parameters={"timeout": 15.0, "response_format": {"type": "json_object"}},
        use_cache=True,
    )

    # Create the code improver agent (in the form of the chain agent) with the subagents required to improve the code
    code_improver = Chain.create(
        name="CodeImprover",
        nodes=[
            CallSubagent(agent=AgentSelector.create(llm)),
            CallSubagent(agent=StepSelector.create(llm), inputs=("AgentSelector",)),
            CallSubagent(agent=StepRewriter.create(llm), inputs=("AgentSelector", "StepSelector")),
        ],
    )

    return code_improver, bad_tape, improver_tape


def main(mode: Literal["run improver", "studio agent", "studio improver"]):
    """
    Main function to run different modes of the tape improver example.

    Parameters:
    mode (Literal["run improver", "studio agent", "studio improver"]):
        The mode in which to run the application.
        - "run improver": Runs the code improver over a bad tape and writes the final tape to a JSON file.
        - "studio improver": Launches the studio Gradio UI with the code improver and improver tape.
        - "studio agent": Launches the studio Gradio UI with a data science agent, bad tape, and environment.

    Raises:
    ValueError: If the final tape does not contain a valid rewrite step in "studio agent" mode.
    AssertionError: If an invalid mode is provided.
    """
    code_improver, bad_tape, improver_tape = make_world()
    if mode == "run improver":
        final_tape = code_improver.run(improver_tape).get_final_tape()
        with open("final_tape.json", "w") as f:
            f.write(final_tape.model_dump_json(indent=2))
    elif mode == "studio improver":
        from tapeagents.studio import Studio

        Studio(code_improver, improver_tape, PrettyRenderer()).launch()
    elif mode == "studio agent":
        data_science_agent, _, env = data_science_make_world()

        def improve_code(tape: Tape):
            improver_start_tape = CodeImproverTape(context=tape, steps=[])
            improver_final_tape = code_improver.run(improver_start_tape).get_final_tape()
            observe_tape(improver_final_tape)
            if not isinstance(rewrite := improver_final_tape[-1], RewriteStep):
                raise ValueError(
                    f"Could not improve tape {tape.metadata.id}"
                    f"For details see tape {improver_final_tape.metadata.id}"
                )
            result = tape.model_copy(
                update=dict(
                    steps=tape.steps[: rewrite.step_index] + [rewrite.new_step],
                    metadata=TapeMetadata(author_tape_id=improver_final_tape.metadata.id),
                )
            )
            return result

        transforms = {"improve_code": improve_code}
        from tapeagents.studio import Studio

        Studio(data_science_agent, bad_tape, make_renderers(), env, transforms).launch()
    else:
        assert False, f"Invalid mode {mode}"


if __name__ == "__main__":
    if len(sys.argv) < 2:
        main("run improver")
    if sys.argv[1] == "agent":
        main("studio agent")
    elif sys.argv[1] == "improver":
        main("studio improver")
    else:
        print("Example of the agent that improves previous bad tape from another agent.")
        print("Usage: python -m examples.tape_improver [agent | improver]")

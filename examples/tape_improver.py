import json
import pathlib
import sys
from typing import Any, Literal

from tapeagents.agent import Agent
from tapeagents.chain import Chain
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
from tapeagents.observe import observe_tape
from tapeagents.rendering import PrettyRenderer
from tapeagents.team import TeamTape
from tapeagents.view import Call, Respond

from examples.data_science import make_renderers
from examples.data_science import make_world as data_science_make_world

MUST_BE_JSON: str = """Output ONLY the JSON in the requested format. Do not output any text before or after JSON. Do not output triple quotes before or after JSON."""

CONTEXT_TAPE_PREFIX: str = """You are observing a history of a team of AI agents working together. We will refer to this history as 'tape'. The tape
consists of steps taken by different agents. The steps may include reasoning thoughts, messages to each other, as well as 
observations coming from the environment. The steps by the agents on the team all have a 'by' attribute with the hierarchical name of the agent. 
As the agent are arranged in an hierarchy, the 'by' attribute also includes of the parents of the agents. Here's the tape:

{context_tape}
"""

SELECT_AGENT_SUFFIX = """You will select the agent that writes code in this organization. 

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

SELECT_STEP_SUFFIX = """Based on the feedback {agent_name} is getting for the code they write 
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

REWRITE_STEP_SUFFIX = """In the previous steps you found that the agent {agent_name} wrote bad code at step {step_index}.
    
    Your reasoning was as follows: {reasoning}.
    
    You will rewrite the step in which the agent wrote bad code. Use the feedback that the agent received 
    later in the tape after Step {step_index}. If you want the user to take some steps before rerunning the code,
    express these steps as an executable code block.
    
    You must respect the schema of the step. Just rewrite the field of the step that contains bad code. 
    Keep the index field and other metadata field.
    
    """


class SelectAgent(Thought):
    kind: str = "select_agent"
    agent_name: str | None
    reason: str


class SelectStep(Thought):
    kind: str = "select_step"
    reasoning: str
    step_index: int | None


class RewriteStep(Action):
    kind: str = "rewrite_step"
    step_index: int
    # code can be written in either Call or Respond messages
    new_step: Call | Respond


class StepParsingError(Action):
    kind: str = "step_parsing_error"
    error: str


CodeImproverTape = Tape[TeamTape, SelectAgent | SelectStep | RewriteStep | FinalStep | Call | Respond]


def improver_tape_view(tape: Tape) -> str:
    data = []
    for index, step in enumerate(tape):
        data.append(step.llm_dict())
        data[-1]["index"] = index
        if isinstance(step, AgentStep):
            data[-1]["metadata"] = {"agent": step.metadata.agent}
    return json.dumps(data, indent=2)


class AgentSelector(Agent):
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
            new_step.metadata.id = '123'
            yield RewriteStep(step_index=select_step.step_index, new_step=new_step)
        except Exception as e:
            yield StepParsingError(error=str(e))


def make_world(llm: LLM | None = None) -> tuple[Agent, Tape, Tape]:
    res_dir = f"{pathlib.Path(__file__).parent.resolve()}/res"
    with open(f"{res_dir}/bad_tape.json", "r") as f:
        bad_tape = TeamTape.model_validate(json.load(f))
    improver_tape = CodeImproverTape(context=bad_tape, steps=[])

    llm = llm or LiteLLM(
        model_name="gpt-4o",
        parameters={"timeout": 15.0, "response_format": {"type": "json_object"}},
        use_cache=True,
    )
    code_improver = Chain.create(
        name="CodeImprover",
        subagents_with_inputs=[
            (AgentSelector.create(llm), ()),
            (StepSelector.create(llm), (-1,)),
            (StepRewriter.create(llm), (-2, -1)),
        ],
    )

    return code_improver, bad_tape, improver_tape


def main(mode: Literal["run improver", "studio agent", "studio improver"]):
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
    match sys.argv[1:]:
        case ["studio", "agent"]:
            main("studio agent")
        case ["studio", "improver"]:
            main("studio improver")
        case ["run", "improver"]:
            main("run improver")
        case _:
            # print usage and exit
            print("Usage: python -m examples.data_science [studio agent] [studio improver]")

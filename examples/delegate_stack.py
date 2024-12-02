from __future__ import annotations

import json
from typing import Annotated, Literal, TypeAlias

from pydantic import Field

from tapeagents.agent import Agent
from tapeagents.core import (
    Action,
    FinalStep,
    Prompt,
    Tape,
    Thought,
)
from tapeagents.llms import LLM, LLMStream, TrainableLLM
from tapeagents.nodes import CallSubagent
from tapeagents.team import Chain
from tapeagents.view import Call, Respond, TapeViewStack

EXAMPLE_TEXT = """I am a text with some verbs like running, jumping, and swimming."""

FIND_VERBS_MESSAGE = """Your job is to find all verbs in the following text:

{text}

You must output JSON like this:

{{
    "verbs": ["verb1", "verb2", ...]
}} 
Do not output any text before or after JSON."""

FIND_NOUNS_MESSAGE = """Your job is to find all nouns in the following text:

{text}

You must output JSON like this:
    
{{
    "nouns": ["noun1", "noun2", ...]
}}

Do not output any text before or after JSON. Use the infinitive form for each verb in your output.
"""

FILTER_IRREGULAR_MESSAGE = """Your job is to filter out irregular verbs from this list of verbs you received below. 

{verbs}

You must output JSON like this:

{{
    "verbs": ["verb1", "verb2", ...]

}}

Do not output any text before or after JSON. Use the infinitive form for each verb in your output.
"""

PRESENT_RESULTS_MESSAGE = """You are a linguist who has done some analysis of the text. 

You found some nouns: {nouns}
And you found some irregular verbs: {irregural_verbs}

Write a nice message discussing the style of the text based on the information you found."""


class InputText(Thought):
    """Thought that the agent should use to pass text as an input to a worker"""

    kind: Literal["input_text"] = "input_text"
    text: str


class AllVerbs(Thought):
    kind: Literal["all_verbs"] = "all_verbs"
    verbs: list[str]


class AllNouns(Thought):
    kind: Literal["all_nouns"] = "all_nouns"
    nouns: list[str]


class IrregularVerbs(Thought):
    kind: Literal["irregular_verbs"] = "irregular_verbs"
    verbs: list[str]


class PresentResults(Action):
    kind: Literal["present_results"] = "present_results"
    content: str


class LastStep(FinalStep):
    kind: Literal["last_step"] = "last_step"


KnownStep: TypeAlias = Annotated[
    InputText | AllVerbs | AllNouns | IrregularVerbs | Call | Respond | LastStep | PresentResults,
    Field(discriminator="kind"),
]
ExampleTape = Tape[str, KnownStep]


class FindVerbs(Agent[ExampleTape]):
    def make_prompt(self, tape: ExampleTape):
        return Prompt.from_user_message(self.templates["default"].format(text=tape.context))

    def generate_steps(self, tape: ExampleTape, llm_stream: LLMStream):
        yield AllVerbs.model_validate_json(llm_stream.get_text())
        yield Respond(copy_output=True)


class FilterIrregular(Agent[ExampleTape]):
    def make_prompt(self, tape: ExampleTape):
        assert isinstance(tape.steps[-1], AllVerbs)
        return Prompt.from_user_message(self.template.format(verbs=json.dumps(tape.steps[-1].verbs)))

    def generate_steps(self, tape: ExampleTape, llm_stream: LLMStream):
        yield IrregularVerbs.model_validate_json(llm_stream.get_text())
        yield Respond(copy_output=True)


class FindNouns(Agent[ExampleTape]):
    def make_prompt(self, tape: ExampleTape):
        return Prompt.from_user_message(self.template.format(text=tape.context))

    def generate_steps(self, tape: ExampleTape, llm_stream: LLMStream):
        yield AllNouns.model_validate_json(llm_stream.get_text())
        yield Respond(copy_output=True)


class Linguist(Chain[ExampleTape]):
    """Analyze the style of the text based on the nouns and irregular verbs that it contains.

    This version shows how you a Chain agent can make its own prompts and generate its own steps.

    """

    @classmethod
    def create(cls, llm: LLM):  # type: ignore
        return super().create(
            llms=llm,
            nodes=[
                CallSubagent(agent=FindNouns.create(llms=llm, templates=FIND_NOUNS_MESSAGE)),
                CallSubagent(
                    agent=Chain.create(
                        name="FindIrregularVerbs",
                        nodes=[
                            CallSubagent(agent=FindVerbs.create(llm, templates=FIND_VERBS_MESSAGE)),
                            CallSubagent(
                                agent=FilterIrregular.create(llm, templates=FILTER_IRREGULAR_MESSAGE),
                                inputs=(-1,),
                            ),
                        ],
                    ),
                ),
            ],
            templates=PRESENT_RESULTS_MESSAGE,
        )

    def make_prompt(self, tape: ExampleTape) -> Prompt:
        state = TapeViewStack.compute(tape)
        if "all_nouns" in state.top.steps_by_kind and "irregular_verbs" in state.top.steps_by_kind:
            (nouns,) = state.top.steps_by_kind["all_nouns"]
            irregular_verbs = state.top.steps_by_kind["irregular_verbs"][0]
            assert isinstance(nouns, AllNouns)
            assert isinstance(irregular_verbs, IrregularVerbs)
            return Prompt.from_user_message(
                self.template.format(
                    nouns=json.dumps(nouns.nouns),
                    irregural_verbs=json.dumps(irregular_verbs.verbs),
                )
            )
        else:
            return Prompt()

    def generate_steps(self, tape: ExampleTape, llm_stream: LLMStream):
        if llm_stream:
            yield PresentResults(content=llm_stream.get_text())
        else:
            yield from super().generate_steps(tape, llm_stream)


class PresentAnalysis(Agent[ExampleTape]):
    def make_prompt(self, tape: ExampleTape):
        assert isinstance(tape.steps[-2], AllNouns)
        assert isinstance(tape.steps[-1], IrregularVerbs)
        return Prompt.from_user_message(
            self.template.format(
                nouns=json.dumps(tape.steps[-2].nouns),
                irregural_verbs=json.dumps(tape.steps[-1].verbs),
            )
        )

    def generate_steps(self, tape: ExampleTape, llm_stream: LLMStream):
        yield PresentResults(content=llm_stream.get_text())
        yield Respond(copy_output=True)


def make_analyze_text_chain(llm: LLM):
    """
    The agent that analyzes the text for nouns and irregular verbs and then presents the results.
    """
    return Chain.create(
        name="Linguist",
        nodes=[
            CallSubagent(agent=FindNouns.create(llms=llm, templates=FIND_NOUNS_MESSAGE)),
            CallSubagent(
                agent=Chain.create(
                    name="FindIrregularVerbs",
                    nodes=[
                        CallSubagent(agent=FindVerbs.create(llm, templates=FIND_VERBS_MESSAGE)),
                        CallSubagent(
                            agent=FilterIrregular.create(llm, templates=FILTER_IRREGULAR_MESSAGE),
                            inputs=(-1,),
                        ),
                    ],
                ),
            ),
            CallSubagent(agent=PresentAnalysis.create(llm, templates=PRESENT_RESULTS_MESSAGE), inputs=(-2, -1)),
        ],
    )


def main():
    llama = TrainableLLM(
        base_url="https://api.together.xyz",
        model_name="meta-llama/Meta-Llama-3-70B-Instruct-Turbo",
        tokenizer_name="meta-llama/Meta-Llama-3-70B-Instruct",
        parameters=dict(temperature=0.7, max_tokens=512),
        use_cache=True,
    )
    with open("llm.json", "w") as f:
        json.dump(llama.model_dump(), f, indent=2)
    tape = ExampleTape(context=EXAMPLE_TEXT)
    agent1 = Linguist.create(llama)
    with open("start_tape.json", "w") as f:
        json.dump(tape.model_dump(), f, indent=2)
    agent1 = Linguist.create(llama)
    tape1 = agent1.run(tape).get_final_tape()
    print(tape1.model_dump_json(indent=2))
    with open("tape1.json", "w") as f:
        json.dump(tape1.model_dump(), f, indent=2)
    agent2 = make_analyze_text_chain(llama)
    tape2 = agent2.run(tape).get_final_tape()
    print(tape2.model_dump_json(indent=2))
    with open("tape2.json", "w") as f:
        json.dump(tape2.model_dump(), f, indent=2)


if __name__ == "__main__":
    main()

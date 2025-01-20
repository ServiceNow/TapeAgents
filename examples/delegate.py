import json
import logging
from typing import Literal

from tapeagents.agent import Agent, AgentEvent
from tapeagents.core import Action, Prompt, Tape, Thought
from tapeagents.llms import LLM, LLMStream, TrainableLLM

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

EXAMPLE_TEXT = """I am a text with some verbs like running, jumping, and swimming."""

FIND_VERBS_MESSAGE = """You job is to find all verbs in the following text:

{text}

You must output JSON like this:

{{
    "verbs": ["verb1", "verb2", ...]
}} 
Do not output any text before or after JSON."""

FILTER_IRREGULAR_MESSAGE = """You job is to filter out irregular verbs from this list of verbs you received below. 

{verbs}

You must output JSON like this:

{{
    "verbs": ["verb1", "verb2", ...]

}}

Do not output any text before or after JSON. Use the infinitive form for each verb in your output.
"""


class AllVerbs(Thought):
    kind: Literal["all_verbs"] = "all_verbs"
    verbs: list[str]


class IrregularVerbs(Action):
    kind: Literal["irregular_verbs"] = "irregular_verbs"
    verbs: list[str]


ExampleTape = Tape[str, AllVerbs | IrregularVerbs]
ExampleEvent = AgentEvent[ExampleTape]


class FindVerbs(Agent[ExampleTape]):
    message_template: str = FIND_VERBS_MESSAGE

    def make_prompt(self, tape: ExampleTape) -> Prompt:
        return Prompt.from_user_message(self.message_template.format(text=tape.context))

    def generate_steps(self, tape: ExampleTape, llm_stream: LLMStream):
        yield AllVerbs.model_validate_json(llm_stream.get_text())


class FilterIrregular(Agent[ExampleTape]):
    message_template: str = FILTER_IRREGULAR_MESSAGE

    def make_prompt(self, tape: ExampleTape) -> Prompt:
        assert isinstance(tape.steps[-1], AllVerbs)
        return Prompt.from_user_message(self.message_template.format(verbs=json.dumps(tape.steps[-1].verbs)))

    def generate_steps(self, tape: ExampleTape, llm_stream: LLMStream):
        yield IrregularVerbs.model_validate_json(llm_stream.get_text())


class FindIrregularVerbs(Agent[ExampleTape]):
    @classmethod
    def create(cls, llm: LLM):  # type: ignore
        return super().create(subagents=[FindVerbs.create(llm), FilterIrregular.create(llm)])

    def delegate(self, tape: ExampleTape) -> Agent[ExampleTape]:
        state = set(step.kind for step in tape.steps)
        if state == set():
            return self.find_subagent("FindVerbs")
        elif state == {"all_verbs"}:
            return self.find_subagent("FilterIrregular")
        else:
            raise ValueError(f"Invalid state {state}")


def try_delegation(llama: TrainableLLM):
    tape = ExampleTape(context=EXAMPLE_TEXT)
    with open("start_tape.json", "w") as f:
        json.dump(tape.model_dump(), f, indent=2)
    agent = FindIrregularVerbs.create(llama)
    tape = agent.run(tape).get_final_tape()
    agent_dump = agent.model_dump()
    print(json.dumps(agent_dump, indent=2))
    print(tape.model_dump_json(indent=2))
    with open("tape.json", "w") as f:
        json.dump(tape.model_dump(), f, indent=2)
    agent_dump["subagents"][0]["llms"]["default"]["parameters"]["temperature"] = 0.444
    another_agent = agent.update(agent_dump)
    print(another_agent.model_dump_json(indent=2))


if __name__ == "__main__":
    try_delegation(
        TrainableLLM(
            base_url="https://api.together.xyz",
            model_name="meta-llama/Meta-Llama-3-70B-Instruct-Turbo",
            parameters=dict(temperature=0.7, max_tokens=512),
        )
    )

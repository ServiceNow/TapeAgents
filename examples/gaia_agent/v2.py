from typing import Any, Callable, Union

from pydantic import Field

from tapeagents.agent import Agent
from tapeagents.core import Prompt, SetNextNode, Tape
from tapeagents.dialog_tape import AssistantStep
from tapeagents.llms import LLM, LLMStream
from tapeagents.utils import FatalError, get_step_schemas_from_union_type

from .agent import GaiaNode
from .prompts import PromptRegistry
from .steps import ListOfFactsThoughtV2, PlanThoughtV2
from .tape import GaiaTape


class GaiaPlainTextNode(GaiaNode):
    tape_view: Callable[[GaiaTape], str] = Field(exclude=True, default=lambda tape: None)

    def get_steps_description(self, tape: GaiaTape, agent: Any) -> str:
        return ""

    def prepare_tape(self, tape: GaiaTape) -> GaiaTape:
        tape = super().prepare_tape(tape)
        tape.steps = [step for step in tape.steps if not isinstance(step, AssistantStep)]
        return tape

    def tape_to_messages(self, tape: GaiaTape, steps_description: str) -> list[dict]:
        tape_view = self.tape_view(tape)
        if tape_view:
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": tape_view},
                {"role": "user", "content": self.guidance},
            ]
        else:
            messages = super().tape_to_messages(tape, steps_description)
        return messages

    def generate_steps(self, agent: Any, tape: GaiaTape, llm_stream: LLMStream):
        cnt = 0
        if self.next_node:
            yield SetNextNode(next_node=self.next_node)
        for event in llm_stream:
            if event.output:
                cnt += 1
                assert event.output.content
                yield AssistantStep(content=event.output.content)
        if not cnt:
            raise FatalError("No completions!")


class Formalize(GaiaNode):
    system_prompt: str = PromptRegistry.formalize_system_prompt
    guidance: str = PromptRegistry.formalize_guidance
    input: str = PromptRegistry.formalize_input
    format: str = PromptRegistry.formalize_format

    def make_prompt(self, agent: Any, tape: Tape) -> Prompt:
        assert isinstance(tape[-1], AssistantStep)
        content: str = tape[-1].content  # type: ignore
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": self.input.format(content=content)},
            {"role": "user", "content": self.guidance},
            {
                "role": "user",
                "content": self.format.format(schema=get_step_schemas_from_union_type(Union[self.agent_step_cls])),
            },
        ]
        return Prompt(messages=messages)


class GaiaPlanner(Agent):
    @classmethod
    def create(
        cls,
        llm: LLM,
    ):
        nodes = [
            GaiaPlainTextNode(
                name="facts_survey",
                system_prompt=PromptRegistry.facts_survey_v2_system,
                guidance=PromptRegistry.facts_survey_v2,
            ),
            Formalize(
                name="formalize_survey",
                agent_step_cls=ListOfFactsThoughtV2,
            ),
            GaiaPlainTextNode(
                name="plan",
                guidance=PromptRegistry.plan_v2,
            ),
            Formalize(
                name="formalize_plan",
                agent_step_cls=PlanThoughtV2,
            ),
        ]
        return super().create(llm, nodes=nodes, max_iterations=2)


class GaiaManager(Agent):
    """
    Follows the plan, manages the execution of the subtasks, reflects on subtask results.
    """

    @classmethod
    def create(
        cls,
        llm: LLM,
    ):
        nodes = [
            GaiaNode(name="start_execution", guidance=PromptRegistry.start_execution),
            GaiaNode(name="default", next_node="default"),
        ]
        return super().create(llm, nodes=nodes, max_iterations=2)

from typing import Any, Union

from tapeagents.agent import Agent, Node
from tapeagents.core import Prompt, Tape
from tapeagents.dialog_tape import AssistantStep
from tapeagents.llms import LLM
from tapeagents.nodes import MonoNode, PlainTextNode
from tapeagents.utils import get_step_schemas_from_union_type

from .agent import GaiaNode
from .prompts import PromptRegistry
from .steps import (
    ActionReflection,
    CurrentPlanStep,
    GaiaAnswer,
    ListOfFactsThoughtV2,
    PlanReflection,
    PlanStepReflection,
    PlanThoughtV2,
)


class Formalize(MonoNode):
    """
    Node that translates plain text response in the last step of the tape into a structured thought of given type.
    """

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


class ChooseAndExecutePlanStep(MonoNode):
    """
    Choose current plan step, add `current_plan_step` step to the tape, call subagent to work on a step
    """

    agent_step_cls: Any = CurrentPlanStep


class ReflectPlanStep(MonoNode):
    """
    Reflects on the plan step success
    """

    agent_step_cls: Any = PlanStepReflection


class NextPlanStep(Node):
    """
    Decides either to go to the next step or to finish the plan
    """

    next_node: str


class ReflectPlan(MonoNode):
    """
    Reflects on the whole plan success
    """

    agent_step_cls: Any = PlanReflection


class Replan(PlainTextNode):
    """
    Produces a new plan based on the reflection of the failed one.
    """

    system_prompt: str = PromptRegistry.system_prompt
    guidance: str = ""


class ProduceAnswer(Formalize):
    """
    Produces the final answer out of the final plan reflection.
    """

    agent_step_cls: Any = GaiaAnswer


class ReflectObservation(PlainTextNode):
    """
    Reflects on the observation, adds it to the tape.
    """

    system_prompt: str = PromptRegistry.system_prompt
    guidance: str = ""


class GaiaPlanner(Agent):
    @classmethod
    def create(
        cls,
        llm: LLM,
    ):
        subagents = [GaiaManager(name="manager")]
        nodes = (
            PlainTextNode(
                name="facts_survey",
                system_prompt=PromptRegistry.facts_survey_v2_system,
                guidance=PromptRegistry.facts_survey_v2,
            ),
            Formalize(name="formalize_survey", agent_step_cls=ListOfFactsThoughtV2),
            PlainTextNode(name="plan", system_prompt=PromptRegistry.system_prompt, guidance=PromptRegistry.plan_v2),
            Formalize(name="formalize_plan", agent_step_cls=PlanThoughtV2, next_agent="manager"),
        )
        return super().create(llm, nodes=nodes, subagents=subagents, max_iterations=2)


class GaiaManager(Agent):
    @classmethod
    def create(
        cls,
        llm: LLM,
    ):
        subagents = [GaiaExecutor()]

        # Yes, it looks like ancient asm code listing with jumps
        nodes = (
            # add `current_plan_step` step to the tape, call subagent to work on a step
            ChooseAndExecutePlanStep(next_agent="GaiaExecutor"),
            # receive the result of the subagent from the last step and reflects on it. add the reflection to the tape
            ReflectPlanStep(),
            # decide either to go to the next step or to finish the plan
            NextPlanStep(next_node="ChooseAndExecutePlanStep"),
            # here we're either executed all the steps successfully or failed on some step, reflects on the whole plan success
            ReflectPlan(next_node="ProduceAnswer"),
            Replan(),  # produce plan based on the reflection of failed one
            # formalize new plan and jump to execution
            Formalize(agent_step_cls=PlanThoughtV2, next_node="ChooseAndExecutePlanStep"),
            ProduceAnswer(),  # produce the final answer
        )
        return super().create(llm, nodes=nodes, subagents=subagents, max_iterations=2)


class GaiaExecutor(Agent):
    """
    Follows the plan, manages the execution of the subtasks, reflects on subtask results.
    """

    @classmethod
    def create(
        cls,
        llm: LLM,
    ):
        nodes = (
            GaiaNode(name="start_execution", guidance=PromptRegistry.start_execution_v2),
            PlainTextNode(name="think", system_prompt=PromptRegistry.system_prompt, guidance=PromptRegistry.todo_next),
            GaiaNode(name="act"),
            ReflectObservation(),
            Formalize(agent_step_cls=ActionReflection, next_node="think"),
        )
        return super().create(llm, nodes=nodes, max_iterations=2)

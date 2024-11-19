import logging
from typing import Any

from tapeagents.agent import Agent, Node
from tapeagents.core import Call, Prompt, ReferenceStep, Step, Tape
from tapeagents.dialog_tape import UserStep
from tapeagents.llms import LLM, LLMStream
from tapeagents.nodes import ControlFlowNode, MonoNode, MonoNodeV2, Return
from tapeagents.utils import get_step_schemas_from_union_type
from tapeagents.view import all_positions, all_steps, first_position, first_step, last_position, last_step

from .prompts import PromptRegistry
from .steps import (
    ActionExecutionFailure,
    ExecutorStep,
    Facts,
    GaiaAction,
    GaiaAnswer,
    GaiaObservation,
    GaiaQuestion,
    PageObservation,
    Plan,
    PlanReflection,
    Subtask,
    SubtaskResult,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ManagerView:
    def __init__(self, tape: Tape):
        last_plan_position = last_position(tape, Plan)
        plan: Plan = tape[last_plan_position]  # type: ignore

        self.steps = {step.number: step for step in plan.plan}
        self.completed_steps = {step.number: step for step in all_steps(tape, SubtaskResult)}
        self.remaining_steps = sorted([n for n in self.steps.keys() if n not in self.completed_steps])
        self.next_step = self.steps[self.remaining_steps[0]] if len(self.remaining_steps) else None

        self.last_subtask_result = last_step(tape, SubtaskResult, allow_none=True)
        self.can_continue = bool(
            self.last_subtask_result and self.last_subtask_result.success and len(self.remaining_steps)
        )
        self.plan_reflection = last_step(tape, PlanReflection, allow_none=True)
        self.success = bool(self.plan_reflection and self.plan_reflection.task_solved)
        self.facts = last_step(tape, Facts, allow_none=True)


class CallManager(Node):
    agent_name: str

    def generate_steps(self, agent: Any, tape: Tape, llm_stream: LLMStream):
        task = first_position(tape, GaiaQuestion)
        plan = last_position(tape, Plan)
        facts = last_position(tape, Facts)
        yield Call(agent_name=self.agent_name)
        yield ReferenceStep(step_number=task)
        yield ReferenceStep(step_number=facts)
        yield ReferenceStep(step_number=plan)


class CallExecutor(Node):
    agent_name: str

    def generate_steps(self, agent: Any, tape: Tape, llm_stream: LLMStream):
        view = ManagerView(tape)
        assert view.next_step, "No remained steps left!"
        prerequisites = []
        for step_number, _ in view.next_step.prerequisites:
            assert step_number in view.completed_steps, f"Prerequisite result {step_number} not found!"
            prerequisites += view.completed_steps[step_number].results

        yield Call(agent_name=self.agent_name)
        yield Subtask(
            number=view.next_step.number,
            name=view.next_step.name,
            description=view.next_step.description,
            prerequisites=prerequisites,
            list_of_tools=view.next_step.list_of_tools,
            expected_results=view.next_step.expected_results,
        )


class ReflectPlan(MonoNodeV2):
    """
    Reflects on the whole plan success
    """

    system_prompt: str = PromptRegistry.system_prompt
    guidance: str = PromptRegistry.reflect_plan_status
    output_cls: Any = PlanReflection

    def tape_to_steps(self, tape: Tape) -> list[Step]:
        steps = super().tape_to_steps(tape)
        view = ManagerView(tape)
        known_facts_str = ""
        if view.facts:
            known_facts_str = "\n".join(view.facts.given_facts + view.facts.found_facts)
        plan_status = PromptRegistry.plan_status.format(
            completed_steps=len(view.completed_steps),
            total_steps=len(view.steps),
            remaining_steps=len(view.remaining_steps),
            facts=known_facts_str,
        )
        steps.append(UserStep(content=plan_status))
        return steps


class UpdateFacts(MonoNodeV2):
    """
    Updates the list of facts based on the subtask results
    """

    system_prompt: str = PromptRegistry.system_prompt
    guidance: str = ""
    output_cls: Any = Facts

    def filter_steps(self, steps: list[Step]) -> list[Step]:
        steps = super().filter_steps(steps)
        all_observation_positions = all_positions(steps, GaiaObservation)
        for position in all_observation_positions[:-1]:
            steps[position] = steps[position].model_copy(update=dict(text=""))
        return steps


class Replan(MonoNodeV2):
    """
    Produces a new plan based on the reflection of the failed one.
    """

    system_prompt: str = PromptRegistry.system_prompt
    guidance: str = PromptRegistry.replan
    output_cls: Any = Plan

    def make_prompt(self, agent: Any, tape: Tape) -> Prompt:
        view = ManagerView(tape)
        plan_text = "\n".join([f"{step.number}. {step.name}\n{step.description}" for step in view.steps.values()])
        plan_result = last_step(tape, PlanReflection)
        failure_text = f"Failed step: {plan_result.failed_step_number}\n{plan_result.failure_overview}"
        self.guidance = PromptRegistry.replan.format(plan=plan_text, failure=failure_text)
        return super().make_prompt(agent, tape)


class ProduceAnswer(MonoNodeV2):
    """
    Produces the final answer out of the final plan reflection.
    """

    system_prompt: str = PromptRegistry.system_prompt
    guidance: str = PromptRegistry.final_answer
    output_cls: Any = GaiaAnswer

    def make_prompt(self, agent: Any, tape: Tape) -> Prompt:
        steps = [
            first_step(tape, GaiaQuestion),
            last_step(tape, Plan),
            last_step(tape, Facts),
            last_step(tape, PlanReflection),
        ]
        steps = [s for s in steps if s]  # remove None
        self.guidance = PromptRegistry.final_answer.format(task=steps[0].content)
        return super().make_prompt(agent, tape.model_copy(update=dict(steps=steps)))


class ReflectObservation(MonoNodeV2):
    system_prompt: str = PromptRegistry.system_prompt
    guidance: str = PromptRegistry.reflect_observation

    def filter_steps(self, steps: list[Step]) -> list[Step]:
        steps = super().filter_steps(steps)
        all_observation_positions = all_positions(steps, GaiaObservation)
        for position in all_observation_positions[:-1]:
            steps[position] = steps[position].model_copy(update=dict(text=""))
        return steps


class Act(MonoNode):
    system_prompt: str = PromptRegistry.system_prompt
    steps_prompt: str = PromptRegistry.allowed_steps_v2.format(schema=get_step_schemas_from_union_type(ExecutorStep))
    agent_step_cls: Any = ExecutorStep

    def prepare_tape(self, tape: Tape, max_chars: int = 200) -> Tape:
        """
        Trim long observations except for the last 3 steps
        """
        tape = super().prepare_tape(tape)  # type: ignore
        steps = []
        for step in tape.steps[:-3]:
            if isinstance(step, PageObservation):
                short_text = f"{step.text[:max_chars]}\n..." if len(step.text) > max_chars else step.text
                new_step = step.model_copy(update=dict(text=short_text))
            elif isinstance(step, ActionExecutionFailure):
                short_error = f"{step.error[:max_chars]}\n..." if len(step.error) > max_chars else step.error
                new_step = step.model_copy(update=dict(error=short_error))
            else:
                new_step = step
            steps.append(new_step)
        trimmed_tape = tape.model_copy(update=dict(steps=steps + tape.steps[-3:]))
        return trimmed_tape


class GaiaPlanner(Agent):
    @classmethod
    def create(
        cls,
        llm: LLM,
    ):
        subagents = [GaiaManager.create(llm)]
        nodes = (
            MonoNodeV2(
                name="FactsSurvey",
                system_prompt=PromptRegistry.facts_survey_v2_system,
                guidance=PromptRegistry.facts_survey_v2,
                output_cls=Facts,
            ),
            MonoNodeV2(
                name="Plan",
                system_prompt=PromptRegistry.system_prompt,
                guidance=PromptRegistry.plan_v2,
                output_cls=Plan,
            ),
            CallManager(agent_name="GaiaManager"),
            # either executed all the steps successfully or failed on some step
            ControlFlowNode(
                name="IsFinished",
                next_node="ProduceAnswer",
                predicate=lambda tape: ManagerView(tape).success,
            ),
            Replan(next_node="CallManager"),
            ProduceAnswer(),  # produce the final answer
        )
        return super().create(llm, nodes=nodes, subagents=subagents, max_iterations=2)


class GaiaManager(Agent):
    @classmethod
    def create(
        cls,
        llm: LLM,
    ):
        subagents = [GaiaExecutor.create(llm)]

        # Yes, it looks like ancient asm code listing with jumps
        nodes = (
            CallExecutor(agent_name="GaiaExecutor"),
            # go to the next step or finish the plan
            ControlFlowNode(
                name="Loop",
                predicate=lambda tape: ManagerView(tape).can_continue,
                next_node="CallExecutor",
            ),
            ReflectPlan(),
            Return(),
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
            MonoNodeV2(
                name="StartExecution",
                system_prompt=PromptRegistry.system_prompt,
                guidance=PromptRegistry.start_execution_v2,
            ),
            Act(),
            ReflectObservation(),
            ControlFlowNode(
                name="ReturnIfFinished",
                predicate=lambda tape: bool(
                    not isinstance(last_step(tape, GaiaAction, allow_none=True), SubtaskResult)
                ),
                next_node="Act",
            ),
            MonoNodeV2(
                name="ReflectSubtask",
                system_prompt=PromptRegistry.system_prompt,
                guidance=PromptRegistry.reflect_subtask,
            ),
            # reflect on the subtask result
            UpdateFacts(),
            Return(),
        )
        return super().create(llm, nodes=nodes, max_iterations=2)

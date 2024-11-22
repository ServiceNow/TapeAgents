import logging
from typing import Any

from pydantic import Field

from tapeagents.agent import Agent, Node
from tapeagents.core import Call, ReferenceStep, Step, Tape
from tapeagents.dialog_tape import UserStep
from tapeagents.llms import LLM, LLMStream
from tapeagents.nodes import ControlFlowNode, Formalize, Return
from tapeagents.utils import get_step_schemas_from_union_type
from tapeagents.view import all_positions, all_steps, first_position, first_step, last_position, last_step

from .agent import GaiaNode
from .prompts import PromptRegistry
from .steps import (
    CoderStep,
    ExecutorStep,
    Facts,
    GaiaAnswer,
    GaiaObservation,
    GaiaQuestion,
    Plan,
    PlanReflection,
    ReasonerStep,
    Subtask,
    SubtaskResult,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ManagerView:
    def __init__(self, tape: Tape):
        self.task = first_step(tape, GaiaQuestion)
        self.plan = last_step(tape, Plan)
        last_subtask = None
        for step in tape:
            if isinstance(step, Subtask):
                last_subtask = step
            if isinstance(step, SubtaskResult) and last_subtask:
                step.number = last_subtask.number
        self.steps = {step.number: step for step in self.plan.plan}
        self.completed_steps = {step.number: step for step in all_steps(tape, SubtaskResult)}
        self.remaining_steps = sorted([n for n in self.steps.keys() if n not in self.completed_steps])
        self.next_step = self.steps[self.remaining_steps[0]] if len(self.remaining_steps) else None

        self.last_subtask_result = last_step(tape, SubtaskResult, allow_none=True)
        self.plan_reflection = last_step(tape, PlanReflection, allow_none=True)
        self.success = bool(self.plan_reflection and self.plan_reflection.task_solved)
        self.can_continue = bool(
            (not self.success)
            and self.last_subtask_result
            and self.last_subtask_result.success
            and len(self.remaining_steps)
        )
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


class CallWorker(Node):
    def generate_steps(self, agent: Any, tape: Tape, llm_stream: LLMStream):
        view = ManagerView(tape)
        assert view.next_step, "No remained steps left!"
        previous_results = view.facts.given_facts
        for p in view.next_step.prerequisites:
            if len(p) == 2:
                step_number, _ = p
            elif view.next_step.number:
                step_number = view.next_step.number - 1
            else:
                continue
            if step_number not in view.completed_steps:
                logger.warning(f"Prerequisite result {step_number} not found!")
                continue
            result = view.completed_steps[step_number]
            previous_results.append(result.llm_dict())

        agent_name = view.next_step.list_of_tools[0] if view.next_step.list_of_tools else "Reasoner"
        assert agent_name in ["Coder", "WebSurfer", "Reasoner"], f"Unknown agent name: {agent_name}"
        yield Call(agent_name=agent_name)
        yield Subtask(
            number=view.next_step.number,
            name=view.next_step.name,
            description=view.next_step.description,
            previous_results=previous_results,
            list_of_tools=view.next_step.list_of_tools,
            expected_results=view.next_step.expected_results,
        )


class GaiaNodeV2(GaiaNode):
    system_prompt: str = PromptRegistry.system_prompt
    steps_prompt: str = PromptRegistry.allowed_steps_v2
    output_cls: Any = Field(exclude=True, default=None)

    def get_steps_description(self, tape: Tape) -> str:
        schema = get_step_schemas_from_union_type(self.output_cls)
        return self.steps_prompt.format(schema=schema)


class ReflectPlanProgress(GaiaNodeV2):
    """
    Reflects on the plan progress
    """

    guidance: str = PromptRegistry.reflect_plan_status
    output_cls: Any = None

    def tape_to_steps(self, tape: Tape) -> list[Step]:
        steps = super().tape_to_steps(tape)
        view = ManagerView(tape)
        known_facts_str = ""
        if view.facts:
            known_facts_str = "\n".join([str(fact) for fact in view.facts.given_facts + view.facts.found_facts])
        plan_status = PromptRegistry.plan_status.format(
            completed_steps=len(view.completed_steps),
            total_steps=len(view.steps),
            remaining_steps=len(view.remaining_steps),
            facts=known_facts_str,
        )
        steps.append(UserStep(content=plan_status))
        return steps


class UpdateFacts(GaiaNodeV2):
    """
    Updates the list of facts based on the subtask results
    """

    guidance: str = ""
    output_cls: Any = Facts

    def tape_view(self, tape: Tape) -> str:
        view = ManagerView(tape)
        return PromptRegistry.facts_survey_update.format(
            task=view.task.content,
            plan=view.plan.llm_view(),
            last_results=view.last_subtask_result.llm_view(),
            facts=view.facts.llm_view(),
        )


class Replan(GaiaNodeV2):
    """
    Produces a new plan based on the reflection of the failed one.
    """

    output_cls: Any = Plan

    def tape_view(self, tape: Tape) -> str:
        view = ManagerView(tape)
        return PromptRegistry.replan.format(
            task=view.task.content,
            plan=view.plan.llm_view(),
            result=view.plan_reflection.llm_view(),
            facts=view.facts.llm_view(),
        )


class Guess(GaiaNodeV2):
    output_cls: Any = None

    def tape_view(self, tape: Tape) -> str:
        view = ManagerView(tape)
        return PromptRegistry.guess.format(
            task=view.task.content,
            plan=view.plan.llm_view(),
            result=view.plan_reflection.llm_view(),
            facts=view.facts.llm_view(),
        )


class ProduceAnswer(GaiaNodeV2):
    """
    Produces the final answer out of the final plan reflection.
    """

    output_cls: Any = None

    def tape_view(self, tape: Tape) -> str:
        view = ManagerView(tape)
        return PromptRegistry.final_answer.format(
            task=view.task.content,
            plan=view.plan.llm_view(),
            result=view.plan_reflection.llm_view(),
            facts=view.facts.llm_view(),
        )


class ReflectObservation(GaiaNodeV2):
    guidance: str = PromptRegistry.reflect_observation

    def tape_to_steps(self, tape: Tape) -> list[Step]:
        steps = super().tape_to_steps(tape)
        all_observation_positions = all_positions(steps, GaiaObservation)
        for position in all_observation_positions[:-1]:  # exclude all but the last observation
            steps[position] = steps[position].model_copy(update=dict(text=""))
        return steps


class GaiaPlanner(Agent):
    @classmethod
    def create(
        cls,
        llm: LLM,
    ):
        max_attempts = 3
        subagents = [GaiaManager.create(llm)]
        nodes = (
            GaiaNodeV2(name="FactsSurvey", guidance=PromptRegistry.facts_survey, output_cls=Facts),
            GaiaNodeV2(name="Plan", guidance=PromptRegistry.plan_v2, output_cls=Plan),
            CallManager(agent_name="GaiaManager"),
            # either executed all the steps successfully or failed on some step
            ControlFlowNode(
                name="IsFinished",
                predicate=lambda tape: ManagerView(tape).success,
                next_node="ProduceAnswer",
            ),
            ControlFlowNode(
                name="NoAttemptsLeft",
                predicate=lambda tape: len(all_steps(tape, Plan)) >= max_attempts,
                next_node="Guess",
            ),
            Replan(next_node="CallManager"),
            Guess(),  # try to guess the answer if the replan attempts are over
            Formalize(output_cls=PlanReflection),
            ProduceAnswer(),
            Formalize(name="Answer", output_cls=GaiaAnswer),
        )
        return super().create(llm, nodes=nodes, subagents=subagents, max_iterations=2)


class GaiaManager(Agent):
    @classmethod
    def create(
        cls,
        llm: LLM,
    ):
        subagents = [WebSurfer.create(llm), Reasoner.create(llm), Coder.create(llm)]

        nodes = (
            CallWorker(),
            UpdateFacts(),
            ReflectPlanProgress(),
            Formalize(output_cls=PlanReflection),
            ControlFlowNode(
                name="Loop",
                predicate=lambda tape: ManagerView(tape).can_continue,
                next_node="CallWorker",
            ),
            Return(),
        )
        return super().create(llm, nodes=nodes, subagents=subagents, max_iterations=2)


class Reasoner(Agent):
    @classmethod
    def create(cls, llm: LLM):
        nodes = [
            GaiaNodeV2(name="Start", guidance=PromptRegistry.reason, output_cls=ReasonerStep),
            GaiaNodeV2(name="Act", output_cls=ReasonerStep),
            ControlFlowNode(
                name="ReturnIfFinished",
                predicate=lambda tape: bool(not isinstance(tape[-1], SubtaskResult)),
                next_node="Act",
            ),
            Return(),
        ]
        return super().create(llm, nodes=nodes, max_iterations=2)


class WebSurfer(Agent):
    @classmethod
    def create(cls, llm: LLM):
        nodes = [
            GaiaNodeV2(name="Start", guidance=PromptRegistry.start_execution_v2, output_cls=ExecutorStep),
            GaiaNodeV2(name="Act", output_cls=ExecutorStep),
            # ReflectObservation(),
            ControlFlowNode(
                name="ReturnIfFinished",
                predicate=lambda tape: bool(not isinstance(tape[-1], SubtaskResult)),
                next_node="Act",
            ),
            Return(),
        ]
        return super().create(llm, nodes=nodes, max_iterations=2)


class Coder(Agent):
    @classmethod
    def create(cls, llm: LLM):
        nodes = [
            GaiaNodeV2(name="Start", guidance=PromptRegistry.start_execution_coder, output_cls=CoderStep),
            GaiaNodeV2(name="Act", output_cls=CoderStep),
            # ReflectObservation(),
            ControlFlowNode(
                name="ReturnIfFinished",
                predicate=lambda tape: bool(not isinstance(tape[-1], SubtaskResult)),
                next_node="Act",
            ),
            Return(),
        ]
        return super().create(llm, nodes=nodes, max_iterations=2)

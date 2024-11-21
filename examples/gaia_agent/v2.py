import logging
from typing import Any

from examples.gaia_agent.agent import GaiaNodeV2
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
    ReasonerStep,
    Reflection,
    Subtask,
    SubtaskResult,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ManagerView:
    def __init__(self, tape: Tape):
        last_plan_position = last_position(tape, Plan)
        plan: Plan = tape[last_plan_position]  # type: ignore
        last_subtask = None
        for step in tape:
            if isinstance(step, Subtask):
                last_subtask = step
            if isinstance(step, SubtaskResult) and last_subtask:
                step.number = last_subtask.number
        self.steps = {step.number: step for step in plan.plan}
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


class CallExecutor(Node):
    agent_name: str

    def generate_steps(self, agent: Any, tape: Tape, llm_stream: LLMStream):
        view = ManagerView(tape)
        assert view.next_step, "No remained steps left!"
        known_facts = []
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
            result = view.completed_steps[step_number].llm_dict()
            known_facts.append(result)
        agent_name = self.agent_name
        if (not view.next_step.list_of_tools) or (
            len(view.next_step.list_of_tools) == 1 and view.next_step.list_of_tools[0].lower() == "reasoner"
        ):
            agent_name = "Reasoner"
        yield Call(agent_name=agent_name)
        yield Subtask(
            number=view.next_step.number,
            name=view.next_step.name,
            description=view.next_step.description,
            known_facts=known_facts,
            list_of_tools=view.next_step.list_of_tools,
            expected_results=view.next_step.expected_results,
        )


class Think(MonoNodeV2):
    system_prompt: str = PromptRegistry.system_prompt
    plaintext_cls: Any = Reflection


class ReflectPlan(Think):
    """
    Reflects on the plan progress
    """

    guidance: str = PromptRegistry.reflect_plan_status

    output_cls: Any = PlanReflection

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


class UpdateFacts(Think):
    """
    Updates the list of facts based on the subtask results
    """

    guidance: str = ""
    output_cls: Any = Facts

    def tape_view(self, tape: Tape) -> str:
        last_results_step = ManagerView(tape).last_subtask_result
        task = first_step(tape, GaiaQuestion)
        plan = last_step(tape, Plan)
        facts = last_step(tape, Facts)
        assert facts, "No facts found!"
        return PromptRegistry.facts_survey_update.format(
            task=task.content, plan=plan.llm_view(), last_results=last_results_step.llm_view(), facts=facts.llm_view()
        )


class Replan(Think):
    """
    Produces a new plan based on the reflection of the failed one.
    """

    guidance: str = PromptRegistry.replan
    output_cls: Any = Plan

    def make_prompt(self, agent: Any, tape: Tape) -> Prompt:
        view = ManagerView(tape)
        plan_text = "\n".join([f"{step.number}. {step.name}\n{step.description}" for step in view.steps.values()])
        plan_result = last_step(tape, PlanReflection)
        failure_text = f"Failed step: {plan_result.failed_step_number}\n{plan_result.failure_overview}"
        self.guidance = PromptRegistry.replan.format(plan=plan_text, failure=failure_text)
        return super().make_prompt(agent, tape)


class ProduceAnswer(Think):
    """
    Produces the final answer out of the final plan reflection.
    """

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


class ReflectObservation(Think):
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
            Think(name="FactsSurvey", guidance=PromptRegistry.facts_survey_v2, output_cls=Facts),
            Think(name="Plan", guidance=PromptRegistry.plan_v2, output_cls=Plan),
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
        subagents = [GaiaOld.create(llm), Reasoner.create(llm)]

        # Yes, it looks like ancient asm code listing with jumps
        nodes = (
            CallExecutor(agent_name="GaiaOld"),
            # go to the next step or finish the plan
            UpdateFacts(),
            ReflectPlan(),
            ControlFlowNode(
                name="Loop",
                predicate=lambda tape: ManagerView(tape).can_continue,
                next_node="CallExecutor",
            ),
            Return(),
        )
        return super().create(llm, nodes=nodes, subagents=subagents, max_iterations=2)


class Reasoner(Agent):
    @classmethod
    def create(cls, llm: LLM):
        nodes = [
            GaiaNodeV2(name="Reason", guidance=PromptRegistry.reason, agent_step_cls=ReasonerStep),
            ControlFlowNode(
                name="ReturnIfFinished",
                predicate=lambda tape: bool(not isinstance(tape[-1], SubtaskResult)),
                next_node="Reason",
            ),
            Return(),
        ]
        return super().create(llm, nodes=nodes, max_iterations=2)


class GaiaOld(Agent):
    @classmethod
    def create(cls, llm: LLM):
        nodes = [
            Think(name="StartExecution", guidance=PromptRegistry.start_execution_v2),
            GaiaNodeV2(name="Act", guidance=PromptRegistry.act),
            ControlFlowNode(
                name="ReturnIfFinished",
                predicate=lambda tape: bool(not isinstance(tape[-1], SubtaskResult)),
                next_node="Act",
            ),
            Return(),
        ]
        return super().create(llm, nodes=nodes, max_iterations=2)


class GaiaExecutor(Agent):
    @classmethod
    def create(cls, llm: LLM):
        nodes = (
            Think(name="StartExecution", guidance=PromptRegistry.start_execution_v2),
            Act(),
            ReflectObservation(),
            ControlFlowNode(
                name="ReturnIfFinished",
                predicate=lambda tape: bool(
                    not isinstance(last_step(tape, GaiaAction, allow_none=True), SubtaskResult)
                ),
                next_node="Act",
            ),
            Think(name="ReflectSubtask", guidance=PromptRegistry.reflect_subtask),
            # reflect on the subtask result
            UpdateFacts(),
            Return(),
        )
        return super().create(llm, nodes=nodes, max_iterations=2)

import logging
from typing import Any

from tapeagents.agent import Agent, Node
from tapeagents.core import Call, Prompt, ReferenceStep, Tape
from tapeagents.dialog_tape import AssistantStep
from tapeagents.llms import LLM, LLMStream
from tapeagents.nodes import ControlFlowNode, MonoNode, Return, ThinkingNode
from tapeagents.utils import get_step_schemas_from_union_type
from tapeagents.view import all_steps, first_position, first_step, last_position, last_step

from .prompts import PromptRegistry
from .steps import (
    ActionReflection,
    ExecutorStep,
    Facts,
    GaiaAnswer,
    GaiaQuestion,
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
        assert last_plan_position is not None, "No plan found!"
        plan: Plan = tape[last_plan_position]  # type: ignore

        self.steps = {step.number: step for step in plan.plan}
        self.completed_steps = {step.number: step for step in all_steps(tape, SubtaskResult)}
        self.remaining_steps = sorted([n for n in self.steps.keys() if n not in self.completed_steps])
        self.next_step = self.steps[self.remaining_steps[0]] if len(self.remaining_steps) else None

        self.last_subtask_result = last_step(tape, SubtaskResult)
        self.can_continue = bool(
            self.last_subtask_result and self.last_subtask_result.success and len(self.remaining_steps)
        )
        self.plan_reflection = last_step(tape, PlanReflection)
        self.success = bool(self.plan_reflection and self.plan_reflection.task_solved)
        fact_ledgers = [step for step in all_steps(tape, Facts) if step]
        if fact_ledgers:
            self.facts = fact_ledgers[-1]
            self.facts.found_facts = [fact for ledger in fact_ledgers for fact in ledger.found_facts]
            self.known_facts = self.facts.given_facts + self.facts.found_facts
            self.known_facts_str = "\n".join([f"- {f}" for f in self.known_facts])
        else:
            self.facts = None
            self.known_facts = []
            self.known_facts_str = "No facts found"


class CallManager(Node):
    agent_name: str

    def generate_steps(self, agent: Any, tape: Tape, llm_stream: LLMStream):
        task = first_position(tape, GaiaQuestion)
        plan = last_position(tape, Plan)
        facts = last_position(tape, Facts)
        assert task is not None, "No task found!"
        assert plan is not None, "No plan found!"
        assert facts is not None, "No facts found!"
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


class ReflectPlan(ThinkingNode):
    """
    Reflects on the whole plan success
    """

    system_prompt: str = PromptRegistry.system_prompt
    guidance: str = PromptRegistry.reflect_plan_status
    output_cls: Any = PlanReflection

    def tape_to_messages(self, tape: Tape) -> list[dict]:
        messages = super().tape_to_messages(tape)
        view = ManagerView(tape)
        plan_status = PromptRegistry.plan_status.format(
            completed_steps=len(view.completed_steps),
            total_steps=len(view.steps),
            remaining_steps=len(view.remaining_steps),
            facts=view.known_facts_str,
        )
        messages = messages[:-1] + [{"role": "user", "content": plan_status}] + messages[-1:]
        return messages


class UpdateFacts(ThinkingNode):
    """
    Updates the list of facts based on the subtask results
    """

    system_prompt: str = PromptRegistry.system_prompt
    guidance: str = ""
    output_cls: Any = Facts

    def tape_view(self, tape: Tape) -> str:
        last_results_step = tape.steps[-4]
        assert isinstance(last_results_step, SubtaskResult), f"wrong subtask result: {last_results_step.kind}"
        last_results_thought = tape.steps[-2]
        assert isinstance(last_results_thought, AssistantStep), f"wrong subtask reflection: {last_results_thought.kind}"
        last_results = f"{last_results_step.llm_view()}\n{last_results_thought.content}"
        start_step = tape[0]
        assert isinstance(start_step, GaiaQuestion)
        facts = last_step(tape, Facts)
        assert facts, "No facts found!"
        return PromptRegistry.facts_survey_update.format(
            task=start_step.content,
            last_results=last_results,
            facts=facts.llm_view(),
        )


class Replan(ThinkingNode):
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
        assert plan_result
        failure_text = f"Failed step: {plan_result.failed_step_number}\n{plan_result.failure_overview}"
        self.guidance = PromptRegistry.replan.format(plan=plan_text, failure=failure_text)
        return super().make_prompt(agent, tape)


class ProduceAnswer(ThinkingNode):
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


class GaiaPlanner(Agent):
    @classmethod
    def create(
        cls,
        llm: LLM,
    ):
        subagents = [GaiaManager.create(llm)]
        nodes = (
            ThinkingNode(
                name="FactsSurvey",
                system_prompt=PromptRegistry.facts_survey_v2_system,
                guidance=PromptRegistry.facts_survey_v2,
                output_cls=Facts,
            ),
            ThinkingNode(
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
            # reflect on the subtask result
            UpdateFacts(),
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
            ThinkingNode(
                name="StartExecution",
                system_prompt=PromptRegistry.system_prompt,
                guidance=PromptRegistry.start_execution_v2,
            ),
            # ----LOOP----
            MonoNode(
                # TODO: 1. unify ThinkingNode and MonoNode. 2. add flag to use function calling for actions instead schema in prompt
                name="Act",
                system_prompt=PromptRegistry.system_prompt,
                steps_prompt=PromptRegistry.allowed_steps_v2.format(
                    schema=get_step_schemas_from_union_type(ExecutorStep)
                ),
                agent_step_cls=ExecutorStep,
            ),
            ControlFlowNode(
                name="ReturnIfFinished",
                predicate=lambda tape: bool(isinstance(tape[-1], SubtaskResult)),
                next_node="ReflectSubtask",
            ),
            ThinkingNode(
                name="ReflectObservation",
                system_prompt=PromptRegistry.system_prompt,
                guidance=PromptRegistry.reflect_observation,
                output_cls=ActionReflection,
            ),
            ThinkingNode(
                name="ProposeNextStep",
                system_prompt=PromptRegistry.system_prompt,
                guidance=PromptRegistry.todo_next,
                next_node="Act",
            ),
            # ----END LOOP----
            ThinkingNode(
                name="ReflectSubtask",
                system_prompt=PromptRegistry.system_prompt,
                guidance=PromptRegistry.reflect_subtask,
            ),
            Return(),
        )
        return super().create(llm, nodes=nodes, max_iterations=2)

import logging
from typing import Any

from tapeagents.agent import Agent, Node
from tapeagents.core import Call, Prompt, Respond, Tape
from tapeagents.llms import LLM, LLMStream
from tapeagents.nodes import ControlFlowNode, FixedStepsNode, MonoNode, ThinkingNode
from tapeagents.utils import get_step_schemas_from_union_type
from tapeagents.view import all_steps, first_position, first_step, last_position, last_step

from .prompts import PromptRegistry
from .steps import (
    ActionReflection,
    ExecutorStep,
    GaiaAnswer,
    GaiaQuestion,
    ListOfFactsThoughtV2,
    PlanReflection,
    PlanThoughtV2,
    PreviousFacts,
    Subtask,
    SubtaskResult,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PlanView:
    def __init__(self, tape: Tape):
        step = first_step(tape, PlanThoughtV2)
        assert step is not None, "No plan found!"
        self.steps = {step.number: step for step in step.plan}
        self.completed_steps = {step.number for step in all_steps(tape, Subtask)}
        self.remaining_steps = sorted(list(self.steps.keys() - self.completed_steps))
        self.next_step = self.steps[self.remaining_steps[0]] if len(self.remaining_steps) else None
        self.last_step = last_step(tape, Subtask)
        self.last_step_result = last_step(tape, SubtaskResult)
        self.can_continue = bool(self.last_step_result and self.last_step_result.success and len(self.remaining_steps))
        self.plan_reflection = last_step(tape, PlanReflection)
        self.success = bool(self.plan_reflection and self.plan_reflection.plan_success)


class CallManager(Node):
    agent_name: str

    def generate_steps(self, agent: Any, tape: Tape, llm_stream: LLMStream):
        task = first_position(tape, GaiaQuestion)
        plan = last_position(tape, PlanThoughtV2)
        facts = last_position(tape, ListOfFactsThoughtV2)
        assert task is not None, "No task found!"
        assert plan is not None, "No plan found!"
        assert facts is not None, "No facts found!"
        yield Call(agent_name=self.agent_name, args=[task, facts, plan])


class CallExecutor(Node):
    agent_name: str

    def generate_steps(self, agent: Any, tape: Tape, llm_stream: LLMStream):
        task = last_position(tape, Subtask)
        facts = last_position(tape, PreviousFacts)
        # TODO: facts distract executor, it starts to execute another subtasks, fix
        assert task is not None, "No task found!"
        assert facts is not None, "No facts found!"
        yield Call(agent_name=self.agent_name, args=[task, facts])


class ChoosePlanStep(Node):
    """
    Choose current plan step, add `current_plan_step` step to the tape, call subagent to work on a step
    """

    def generate_steps(self, agent: Any, tape: Tape, llm_stream: LLMStream):
        plan = PlanView(tape)
        assert plan.next_step, "No remained steps left!"
        step = Subtask(
            number=plan.next_step.number,
            name=plan.next_step.name,
            description=plan.next_step.description,
        )
        logger.info(f"Choosing step plan {step.number}:\n{step.llm_view()}")
        yield step


class ReflectPlan(ThinkingNode):
    """
    Reflects on the whole plan success
    """

    system_prompt: str = PromptRegistry.system_prompt
    guidance: str = PromptRegistry.reflect_plan_status
    output_cls: Any = PlanReflection

    def tape_to_messages(self, tape: Tape) -> list[dict]:
        messages = super().tape_to_messages(tape)
        plan = PlanView(tape)
        plan_status = PromptRegistry.plan_status.format(
            completed_steps=len(plan.completed_steps),
            total_steps=len(plan.steps),
            remained_steps=len(plan.remaining_steps),
        )
        messages = messages[:-1] + [{"role": "user", "content": plan_status}] + messages[-1:]
        return messages


class FactSurveyUpdate(ThinkingNode):
    """
    Updates the list of facts based on the subtask results
    """

    system_prompt: str = PromptRegistry.system_prompt
    guidance: str = PromptRegistry.facts_survey_update
    output_cls: Any = ListOfFactsThoughtV2

    def tape_to_messages(self, tape: Tape) -> list[dict]:
        messages = super().tape_to_messages(tape)
        start_step = tape[0]
        assert isinstance(start_step, GaiaQuestion)
        facts_step = first_step(tape, ListOfFactsThoughtV2)
        assert facts_step, "No facts found!"
        assert isinstance(start_step, GaiaQuestion)
        messages[-1]["content"] = PromptRegistry.facts_survey_update.format(
            task=start_step.content,
            available="\n".join(facts_step.available_facts),
            lookup="\n".join(facts_step.facts_to_lookup),
            derive="\n".join(facts_step.facts_to_derive),
            guesses="\n".join(facts_step.educated_guesses),
        )
        return messages


class Replan(ThinkingNode):
    """
    Produces a new plan based on the reflection of the failed one.
    """

    system_prompt: str = PromptRegistry.system_prompt
    guidance: str = PromptRegistry.replan
    output_cls: Any = PlanThoughtV2
    next_node: str = "ChoosePlanStep"

    def make_prompt(self, agent: Any, tape: Tape) -> Prompt:
        plan = PlanView(tape)
        plan_text = "\n".join([f"{step.number}. {step.name}\n{step.description}" for step in plan.steps.values()])
        plan_result = last_step(tape, PlanReflection)
        assert plan_result
        failure_text = f"Failed step: {plan_result.failed_step_number}\n{plan_result.failed_step_overview}"
        self.guidance = PromptRegistry.replan.format(plan=plan_text, failure=failure_text)
        return super().make_prompt(agent, tape)


class ChooseSubtaskFacts(ThinkingNode):
    system_prompt: str = PromptRegistry.system_prompt
    guidance: str = PromptRegistry.choose_facts
    output_cls: Any = PreviousFacts

    def tape_view(self, tape: Tape) -> str:
        facts = last_step(tape, ListOfFactsThoughtV2)
        subtask = last_step(tape, Subtask)
        assert facts, "No facts found!"
        assert subtask, "No subtask found!"
        return PromptRegistry.current_facts.format(subtask=subtask, facts=facts)


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
            last_step(tape, PlanThoughtV2),
            last_step(tape, ListOfFactsThoughtV2),
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
                output_cls=ListOfFactsThoughtV2,
            ),
            ThinkingNode(
                name="Plan",
                system_prompt=PromptRegistry.system_prompt,
                guidance=PromptRegistry.plan_v2,
                output_cls=PlanThoughtV2,
            ),
            CallManager(agent_name="GaiaManager"),
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
            ChoosePlanStep(),  # adds subtask with the current plan step to the tape
            ChooseSubtaskFacts(),  # choose facts relevant for the subtask
            CallExecutor(agent_name="GaiaExecutor"),
            # reflect on the subtask result
            FactSurveyUpdate(),
            # go to the next step or finish the plan
            ControlFlowNode(
                name="Loop",
                next_node="ChoosePlanStep",
                predicate=lambda tape: PlanView(tape).can_continue,
            ),
            ReflectPlan(),
            # either executed all the steps successfully or failed on some step
            ControlFlowNode(
                name="IsFinished",
                next_node="ProduceAnswer",
                predicate=lambda tape: PlanView(tape).success,
            ),
            Replan(),
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
            ThinkingNode(
                name="StartExecution",
                system_prompt=PromptRegistry.system_prompt,
                guidance=PromptRegistry.start_execution_v2,
            ),
            # ----LOOP----
            MonoNode(
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
            FixedStepsNode(name="Return", steps=[Respond(copy_output=True)]),
        )
        return super().create(llm, nodes=nodes, max_iterations=2)

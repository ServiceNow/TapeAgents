import logging
from typing import Any, Union

from tapeagents.agent import Agent, Node
from tapeagents.core import Call, Prompt, Respond, Tape
from tapeagents.dialog_tape import AssistantStep
from tapeagents.llms import LLM, LLMStream
from tapeagents.nodes import ConditionalNode, ControlFlowNode, MonoNode, ThinkingNode
from tapeagents.utils import get_step_schemas_from_union_type
from tapeagents.view import all_steps, first_position, first_step, last_position, last_step

from .prompts import PromptRegistry
from .steps import (
    ActionReflection,
    ExecutorStep,
    FinishSubtask,
    GaiaAnswer,
    GaiaQuestion,
    ListOfFactsThoughtV2,
    PlanReflection,
    PlanStepReflection,
    PlanThoughtV2,
    Subtask,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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


class PlanView:
    def __init__(self, tape: Tape):
        step = first_step(tape, PlanThoughtV2)
        assert step is not None, "No plan found!"
        self.steps = {step.number: step for step in step.plan}
        self.completed_steps = {step.number for step in all_steps(tape, Subtask)}
        self.remaining_steps = sorted(list(self.steps.keys() - self.completed_steps))
        self.next_step = self.steps[self.remaining_steps[0]] if len(self.remaining_steps) else None
        self.last_step = last_step(tape, Subtask)
        self.last_step_result = last_step(tape, PlanStepReflection)
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
        facts = last_position(tape, ListOfFactsThoughtV2)
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

    def tape_to_messages(self, tape: Tape) -> list[dict]:
        messages = super().tape_to_messages(tape)
        start_step = tape[0]
        assert isinstance(start_step, GaiaQuestion)
        facts_step = first_step(tape, ListOfFactsThoughtV2)
        assert facts_step, "No facts found!"
        assert isinstance(start_step, GaiaQuestion)
        messages[-1]["content"] = PromptRegistry.facts_survey_update.format(
            task=start_step.content,
            given=self.facts_list(facts_step.given_facts),
            lookup=self.facts_list(facts_step.facts_to_lookup),
            derive=self.facts_list(facts_step.facts_to_derive),
            guesses=self.facts_list(facts_step.educated_guesses),
        )
        return messages

    def facts_list(self, facts: list) -> str:
        return "\n".join([f" - {fact.name}\n{fact.description}" for fact in facts])


class Replan(ThinkingNode):
    """
    Produces a new plan based on the reflection of the failed one.
    """

    system_prompt: str = PromptRegistry.system_prompt
    guidance: str = PromptRegistry.replan

    def make_prompt(self, agent: Any, tape: Tape) -> Prompt:
        plan = PlanView(tape)
        plan_text = "\n".join([f"{step.number}. {step.name}\n{step.description}" for step in plan.steps.values()])
        plan_result = last_step(tape, PlanReflection)
        assert plan_result
        failure_text = f"Failed step: {plan_result.failed_step_number}\n{plan_result.failed_step_overview}"
        self.guidance = PromptRegistry.replan.format(plan=plan_text, failure=failure_text)
        return super().make_prompt(agent, tape)


class ProduceAnswer(Formalize):
    """
    Produces the final answer out of the final plan reflection.
    """

    agent_step_cls: Any = GaiaAnswer

    def make_prompt(self, agent: Any, tape: Tape) -> Prompt:
        start_step = tape[0]
        assert isinstance(start_step, GaiaQuestion)
        self.guidance = PromptRegistry.final_answer.format(task=start_step.content)
        return super().make_prompt(agent, tape)


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
            ),
            Formalize(name="FormalizeSurvey", agent_step_cls=ListOfFactsThoughtV2),
            ThinkingNode(name="Plan", system_prompt=PromptRegistry.system_prompt, guidance=PromptRegistry.plan_v2),
            Formalize(name="FormalizePlan", agent_step_cls=PlanThoughtV2),
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
            # add `current_plan_step` step to the tape, call subagent to work on a step
            ChoosePlanStep(),
            CallExecutor(agent_name="GaiaExecutor"),
            # receive the result of the subagent from the last step and reflect on it
            ThinkingNode(
                name="ReflectPlanStep",
                system_prompt=PromptRegistry.system_prompt,
                guidance=PromptRegistry.reflect_plan_step_result,
            ),
            Formalize(name="FormalizePlanStepReflection", agent_step_cls=PlanStepReflection),
            FactSurveyUpdate(),
            Formalize(name="FormalizeSurvey", agent_step_cls=ListOfFactsThoughtV2),
            # go to the next step or finish the plan
            ControlFlowNode(
                name="Loop",
                next_node="ChooseAndExecutePlanStep",
                predicate=lambda tape: PlanView(tape).can_continue,
            ),
            ReflectPlan(),
            Formalize(name="FormalizePlanReflection", agent_step_cls=PlanReflection),
            # either executed all the steps successfully or failed on some step
            ControlFlowNode(
                name="IsFinished",
                next_node="ProduceAnswer",
                predicate=lambda tape: PlanView(tape).success,
            ),
            Replan(),
            Formalize(name="FormalizePlan", agent_step_cls=PlanThoughtV2, next_node="ChooseAndExecutePlanStep"),
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
            MonoNode(
                name="Act",
                system_prompt=PromptRegistry.system_prompt,
                steps_prompt=PromptRegistry.allowed_steps_v2.format(
                    schema=get_step_schemas_from_union_type(ExecutorStep)
                ),
                agent_step_cls=ExecutorStep,
            ),
            ConditionalNode(
                name="ReturnIfFinished",
                predicate=lambda tape: bool(last_step(tape, FinishSubtask)),
                steps=[Respond(copy_output=True)],
            ),
            ThinkingNode(
                name="ReflectObservation",
                system_prompt=PromptRegistry.system_prompt,
                guidance=PromptRegistry.reflect_observation,
            ),
            Formalize(name="FormalizeReflectObservation", agent_step_cls=ActionReflection),
            ThinkingNode(
                name="Todo",
                system_prompt=PromptRegistry.system_prompt,
                guidance=PromptRegistry.todo_next,
                next_node="Act",
            ),
        )
        return super().create(llm, nodes=nodes, max_iterations=2)

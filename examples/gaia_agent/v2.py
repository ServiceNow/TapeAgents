from typing import Any, Union

from tapeagents.agent import Agent, Node
from tapeagents.core import Call, Prompt, Respond, SetNextNode, Tape
from tapeagents.dialog_tape import AssistantStep
from tapeagents.llms import LLM, LLMStream
from tapeagents.nodes import ControlFlowNode, MonoNode, ThinkingNode
from tapeagents.utils import get_step_schemas_from_union_type
from tapeagents.view import all_steps, first_step, last_step

from .agent import GaiaNode
from .prompts import PromptRegistry
from .steps import (
    ActionReflection,
    CurrentPlanStep,
    ExecutorStep,
    FactSchema,
    FinishSubtask,
    GaiaAnswer,
    GaiaQuestion,
    ListOfFactsThoughtV2,
    PlanReflection,
    PlanStepReflection,
    PlanThoughtV2,
    ReasoningThought,
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


class PlanView:
    def __init__(self, tape: Tape):
        step = first_step(tape, PlanThoughtV2)
        assert step is not None, "No plan found!"
        self.steps = {step.number: step for step in step.plan}
        self.completed_steps = {step.number for step in all_steps(tape, CurrentPlanStep)}
        self.remaining_steps = sorted(list(self.steps.keys() - self.completed_steps))
        self.next_step = self.steps[self.remaining_steps[0]] if len(self.remaining_steps) else None
        self.last_step = last_step(tape, CurrentPlanStep)
        self.last_step_result = last_step(tape, PlanStepReflection)
        self.can_continue = bool(self.last_step_result and self.last_step_result.success and len(self.remaining_steps))
        self.plan_reflection = last_step(tape, PlanReflection)
        self.success = bool(self.plan_reflection and self.plan_reflection.plan_success)


class ChooseAndExecutePlanStep(Node):
    """
    Choose current plan step, add `current_plan_step` step to the tape, call subagent to work on a step
    """

    next_agent: str

    def generate_steps(self, tape: Tape, **kwargs):
        plan = PlanView(tape)
        assert plan.next_step, "No remained steps left!"
        step = CurrentPlanStep(
            number=plan.next_step.number,
            name=plan.next_step.name,
            description=plan.next_step.description,
        )
        yield step
        yield Call(agent_name=self.next_agent, task=step.llm_dict())


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


class Act(MonoNode):
    """
    Act or return to the upper level agent if finished.
    """

    system_prompt: str = PromptRegistry.system_prompt
    agent_step_cls: Any = ExecutorStep

    def generate_steps(self, agent: Any, tape: Tape, llm_stream: LLMStream):
        for step in super().generate_steps(agent, tape, llm_stream):
            yield step
            if isinstance(step, FinishSubtask):
                yield Respond(copy_output=True)
                break


class GaiaPlanner(Agent):
    @classmethod
    def create(
        cls,
        llm: LLM,
    ):
        subagents = [GaiaManager.create(llm)]
        nodes = (
            ThinkingNode(
                name="facts_survey",
                system_prompt=PromptRegistry.facts_survey_v2_system,
                guidance=PromptRegistry.facts_survey_v2,
            ),
            Formalize(name="formalize_survey", agent_step_cls=ListOfFactsThoughtV2),
            ThinkingNode(name="plan", system_prompt=PromptRegistry.system_prompt, guidance=PromptRegistry.plan_v2),
            Formalize(name="formalize_plan", agent_step_cls=PlanThoughtV2, next_agent="GaiaManager"),
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
            ChooseAndExecutePlanStep(next_agent="GaiaExecutor"),
            # receive the result of the subagent from the last step and reflect on it
            ThinkingNode(
                name="reflect_plan_step",
                system_prompt=PromptRegistry.system_prompt,
                guidance=PromptRegistry.reflect_plan_step_result,
            ),
            Formalize(name="formalize_plan_step_reflection", agent_step_cls=PlanStepReflection),
            FactSurveyUpdate(),
            Formalize(name="formalize_survey", agent_step_cls=ListOfFactsThoughtV2),
            # go to the next step or finish the plan
            ControlFlowNode(
                name="loop",
                next_node="ChooseAndExecutePlanStep",
                predicate=lambda tape: PlanView(tape).can_continue,
            ),
            ReflectPlan(),
            Formalize(name="formalize_plan_reflection", agent_step_cls=PlanReflection),
            # either executed all the steps successfully or failed on some step
            ControlFlowNode(
                name="is_finished",
                next_node="ProduceAnswer",
                predicate=lambda tape: PlanView(tape).success,
            ),
            Replan(),
            Formalize(name="formalize_plan", agent_step_cls=PlanThoughtV2, next_node="ChooseAndExecutePlanStep"),
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
                name="start_execution",
                system_prompt=PromptRegistry.system_prompt,
                guidance=PromptRegistry.start_execution_v2,
            ),
            Formalize(agent_step_cls=ReasoningThought),
            Act(),
            ThinkingNode(
                name="reflect_observation",
                system_prompt=PromptRegistry.system_prompt,
                guidance=PromptRegistry.reflect_observation,
            ),
            Formalize(agent_step_cls=ActionReflection),
            ThinkingNode(name="think", system_prompt=PromptRegistry.system_prompt, guidance=PromptRegistry.todo_next),
            Formalize(agent_step_cls=ReasoningThought, next_node="Act"),
        )
        return super().create(llm, nodes=nodes, max_iterations=2)

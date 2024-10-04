import logging
from enum import Enum
from typing import Any

from tapeagents.core import Step
from tapeagents.guided_agent import GuidanceNode, GuidedAgent
from tapeagents.llms import LLM

from .prompts import PromptRegistry
from .steps import (
    ActionExecutionFailure,
    FinishSubtask,
    GaiaAgentStep,
    GaiaQuestion,
    ListOfFactsThought,
    PageObservation,
    PlanThought,
    PreviousFactsObservation,
    PythonCodeAction,
    SourcesThought,
    UseCalculatorAction,
    get_allowed_steps,
)
from .tape import GaiaTape

logger = logging.getLogger(__name__)


class PlanningMode(str, Enum):
    simple = "simple"
    facts_and_sources = "facts_and_sources"
    multiplan = "multiplan"
    replan_after_sources = "replan_after_sources"


class GaiaNode(GuidanceNode):
    system_prompt: str = PromptRegistry.system_prompt
    steps_prompt: str = PromptRegistry.allowed_steps
    start_step_cls: Any = GaiaQuestion
    agent_step_cls: Any = GaiaAgentStep

    def get_steps_description(self, tape: GaiaTape, agent: Any) -> str:
        """
        Allow different subset of steps based on the agent's configuration
        """
        plan_thoughts = not tape.has_fact_schemas()
        allowed_steps = get_allowed_steps(agent.short_steps, agent.subtasks, plan_thoughts)
        return self.steps_prompt.format(allowed_steps=allowed_steps)

    def prepare_tape(self, tape: GaiaTape, max_chars: int = 200) -> GaiaTape:
        """
        Trim long observations except for the last 3 steps
        """
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

    def postprocess_step(self, tape: GaiaTape, new_steps: list[Step], step: Step) -> Step:
        if isinstance(step, ListOfFactsThought):
            # remove empty facts produced by the model
            step.given_facts = [fact for fact in step.given_facts if fact.value is not None and fact.value != ""]
        elif isinstance(step, (UseCalculatorAction, PythonCodeAction)):
            # if calculator or code action is used, add the facts to the action call
            step.facts = tape.model_copy(update=dict(steps=tape.steps + new_steps)).facts()
        return step


class GaiaAgent(GuidedAgent):
    subtasks: bool
    short_steps: bool

    @classmethod
    def create(
        cls,
        llm: LLM,
        planning_mode: PlanningMode = PlanningMode.simple,
        subtasks: bool = False,
        short_steps: bool = False,
    ):
        nodes = cls.prepare_guidance(planning_mode, subtasks)
        return super().create(llm, nodes=nodes, max_iterations=2, subtasks=subtasks, short_steps=short_steps)

    @classmethod
    def prepare_guidance(cls, planning_mode: PlanningMode, subtasks: bool) -> list[GaiaNode]:
        """
        Prepare guidance nodes based on the planning mode and subtasks flag
        """
        guidance_nodes = []
        if planning_mode == PlanningMode.simple:
            guidance_nodes = [
                GaiaNode(name="plan", trigger_step="question", guidance=PromptRegistry.plan),
                GaiaNode(name="facts_survey", trigger_step="plan_thought", guidance=PromptRegistry.facts_survey),
                GaiaNode(
                    name="start_execution",
                    trigger_step="list_of_facts_thought",
                    guidance=PromptRegistry.start_execution,
                ),
            ]
        elif planning_mode == PlanningMode.facts_and_sources:
            guidance_nodes = [
                GaiaNode(name="plan", trigger_step="question", guidance=PromptRegistry.plan),
                GaiaNode(name="facts_survey", trigger_step="draft_plans_thought", guidance=PromptRegistry.facts_survey),
                GaiaNode(
                    name="sources_plan", trigger_step="list_of_facts_thought", guidance=PromptRegistry.sources_plan
                ),
                GaiaNode(
                    name="start_execution", trigger_step="sources_thought", guidance=PromptRegistry.start_execution
                ),
            ]
        elif planning_mode == PlanningMode.multiplan:
            guidance_nodes = [
                GaiaNode(name="plan", trigger_step="question", guidance=PromptRegistry.plan3),
                GaiaNode(name="facts_survey", trigger_step="draft_plans_thought", guidance=PromptRegistry.facts_survey),
                GaiaNode(
                    name="sources_plan", trigger_step="list_of_facts_thought", guidance=PromptRegistry.sources_plan
                ),
                GaiaNode(
                    name="start_execution", trigger_step="sources_thought", guidance=PromptRegistry.start_execution
                ),
            ]
        elif planning_mode == PlanningMode.replan_after_sources:
            guidance_nodes = [
                GaiaNode(name="plan", trigger_step="question", guidance=PromptRegistry.plan),
                GaiaNode(name="facts_survey", trigger_step="plan_thought", guidance=PromptRegistry.facts_survey),
                GaiaNode(name="better_plan", trigger_step="list_of_facts_thought", guidance=PromptRegistry.better_plan),
                GaiaNode(
                    name="start_execution", trigger_step="sources_thought", guidance=PromptRegistry.start_execution
                ),
            ]
        else:
            raise ValueError(f"Unknown planning mode: {planning_mode}")
        if subtasks:
            guidance_nodes.append(
                GaiaNode(
                    name="check_subtask_finished",
                    trigger_step="calculation_result_observation",
                    guidance=PromptRegistry.is_subtask_finished,
                )
            )
        guidance_nodes.append(GaiaNode(name="default", trigger_step="default", guidance=""))
        return guidance_nodes

    def trim_tape(self, tape: GaiaTape) -> GaiaTape:
        """
        Make tape shorter to fit llm context size limits
        """
        for i, step in enumerate(tape):
            logger.info(f"{i}: {step.__class__.__name__} {self.llm.count_tokens(step.llm_view())} tokens")
        finish_subtask_positions = [i for i, step in enumerate(tape) if isinstance(step, FinishSubtask)]
        # trim either after last finished subtask or at 2/3 of the tape
        summarization_border = (finish_subtask_positions[-1] + 1) if finish_subtask_positions else int(len(tape) * 0.66)
        short_tape = tape.model_copy(update=dict(steps=[]))
        pre_tape: GaiaTape = tape[:summarization_border]  # type: ignore
        for step in pre_tape.steps:
            if isinstance(step, (GaiaQuestion, PlanThought, SourcesThought, ListOfFactsThought)):
                short_tape.steps.append(step)
        short_tape.steps.append(PreviousFactsObservation(facts=pre_tape.facts()))
        for step in tape.steps[summarization_border:]:
            short_tape.steps.append(step)
        logger.info(f"Tape reduced from {len(tape)} to {len(short_tape)} steps")
        return short_tape

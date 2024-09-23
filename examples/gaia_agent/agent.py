import logging
from enum import Enum
from typing import Any

from tapeagents.core import Step
from tapeagents.guided_agent import GuidedAgent

from .prompts import TEMPLATES, PromptRegistry
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


class GaiaAgent(GuidedAgent):
    templates: dict[str, str] = TEMPLATES
    max_iterations: int = 2
    planning_mode: str = PlanningMode.simple
    short_steps: bool = False
    subtasks: bool = False
    _start_step_cls: Any = GaiaQuestion
    _agent_step_cls: Any = GaiaAgentStep
    
    def prepare_guidance_prompts(self):
        guidance_prompts = {}
        if self.planning_mode == PlanningMode.simple:
            guidance_prompts = {
                "question": PromptRegistry.plan,
                "plan_thought": PromptRegistry.facts_survey,
                "list_of_facts_thought": PromptRegistry.start_execution,
            }
        elif self.planning_mode == PlanningMode.facts_and_sources:
            guidance_prompts = {
                "question": PromptRegistry.plan,
                "draft_plans_thought": PromptRegistry.facts_survey,
                "list_of_facts_thought": PromptRegistry.sources_plan,
                "sources_thought": PromptRegistry.start_execution,
            }
        elif self.planning_mode == PlanningMode.multiplan:
            guidance_prompts = {
                "question": PromptRegistry.plan3,
                "draft_plans_thought": PromptRegistry.facts_survey,
                "list_of_facts_thought": PromptRegistry.sources_plan,
                "sources_thought": PromptRegistry.start_execution,
            }
        elif self.planning_mode == PlanningMode.replan_after_sources:
            guidance_prompts = {
                "question": PromptRegistry.plan3,
                "draft_plans_thought": PromptRegistry.facts_survey,
                "list_of_facts_thought": PromptRegistry.sources_plan,
                "sources_thought": PromptRegistry.better_plan,
                "plan_thought": PromptRegistry.start_execution,
            }
        if self.subtasks:
            guidance_prompts["calculation_result_observation"] = PromptRegistry.is_subtask_finished
        self.templates |= guidance_prompts

    def get_steps_description(self, tape: GaiaTape) -> str:
        add_plan_thoughts = not tape.has_fact_schemas()
        allowed_steps = get_allowed_steps(self.short_steps, self.subtasks, add_plan_thoughts)
        return self.templates["allowed_steps"].format(allowed_steps=allowed_steps)

    def prepare_tape(self, tape: GaiaTape, max_chars: int = 200) -> GaiaTape:
        """
        Trim all read results except in the last 3 steps
        """
        self.prepare_guidance_prompts()
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

    def postprocess_step(self, tape: GaiaTape, new_steps: list[Step], step: Step) -> Step:
        if isinstance(step, ListOfFactsThought):
            # remove empty facts produced by the model
            step.given_facts = [fact for fact in step.given_facts if fact.value is not None and fact.value != ""]
        elif isinstance(step, (UseCalculatorAction, PythonCodeAction)):
            # if calculator or code action is used, add the facts to the action call
            step.facts = tape.model_copy(update=dict(steps=tape.steps + new_steps)).facts()
        return step

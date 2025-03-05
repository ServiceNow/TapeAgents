from itertools import takewhile
from typing import Any, Generator

from pydantic import Field

from tapeagents.agent import Agent, Node
from tapeagents.core import Action, AgentStep, LLMOutputParsingFailureAction, PartialStep, SetNextNode, Step, StopStep
from tapeagents.llms import LLM, LLMStream
from tapeagents.nodes import MonoNode
from tapeagents.tools.simple_browser import PageObservation
from tapeagents.utils import FatalError

from .prompts import PromptRegistry
from .steps import (
    WebAgentStep,
    WebStep,
    WebTape,
)


class WebNode(MonoNode):
    max_retries: int = 3
    max_same_action: int = 4
    current_retries: int = 0

    # def trim_tape(self, tape: WebTape) -> WebTape:
    #     """
    #     Trim all page observations except the last two.

    #     Args:
    #         tape (Tape): The tape object to be trimmed.

    #     Returns:
    #         Tape: The trimmed tape object.
    #     """
    #     tape = super().prepare_tape(tape)  # type: ignore
    #     page_positions = [i for i, step in enumerate(tape.steps) if isinstance(step, PageObservation)]
    #     if len(page_positions) < 2:
    #         return tape
    #     prev_page_position = page_positions[-2]
    #     steps = []
    #     for step in tape.steps[:prev_page_position]:
    #         if isinstance(step, PageObservation):
    #             short_text = f"{step.text[:max_chars]}\n..." if len(step.text) > max_chars else step.text
    #             new_step = step.model_copy(update=dict(text=short_text))
    #         else:
    #             new_step = step
    #         steps.append(new_step)
    #     trimmed_tape = tape.model_copy(update=dict(steps=steps + tape.steps[prev_page_position:]))
    #     return trimmed_tape

    def tape_to_messages(self, tape: WebTape, steps_description: str) -> list[dict]:
        """
        Converts a Tape object and steps description into a list of messages for LLM conversation.

        Modifications from the original MonoNode:
        - If the last n steps are LLMOutputParsingFailureAction, put the guidance before the error steps
          and add a new guidance message that asks to retry.
        - Otherwise, remove all LLMOutputParsingFailureAction steps and use the default behavior on the cleaned tape.

        Args:
            tape (Tape): A Tape object containing conversation steps.
            steps_description (str): A description of the conversation steps.

        Returns:
            list[dict]: A list of dictionaries representing the conversation messages.
                       Each dictionary contains 'role' and 'content' keys.
                       Roles can be 'system', 'user', or 'assistant'.
                       The system prompt is always the first message.
                       If steps_description is provided, it's added as a user message.
                       Messages from tape are added with roles based on step type.
                       If guidance exists, it's added as the final user message.
        """
        if len(tape.steps) > 0 and isinstance(tape.steps[-1], LLMOutputParsingFailureAction):
            messages: list[dict] = []
            if self.system_prompt:
                messages.append({"role": "system", "content": self.system_prompt})
            if steps_description:
                messages.append({"role": "user", "content": steps_description})
            # ignore the last n consecutive parsing error steps for now, will add them later
            n_parsing_errors = 0
            for step in reversed(tape.steps):
                if isinstance(step, LLMOutputParsingFailureAction):
                    n_parsing_errors += 1
                else:
                    break
            for step in tape.steps[:-n_parsing_errors]:
                role = "assistant" if isinstance(step, AgentStep) else "user"
                messages.append({"role": role, "content": step.llm_view()})
            # add the initial guidance message
            if self.guidance:
                messages.append({"role": "user", "content": self.guidance})
            # add the last n parsing error steps and the new guidance message that asks to retry
            for step in tape.steps[-n_parsing_errors:]:
                messages.append({"role": "assistant", "content": step.llm_view()})
                messages.append({"role": "user", "content": "You made a mistake. Look at your generation (in 'llm_output') and the error message (in 'error') and please try again."})
            return messages
        else:
            steps_without_parsing_errors = [step for step in tape.steps if not isinstance(step, LLMOutputParsingFailureAction)]
            cleaned_tape = tape.model_copy(update=dict(steps=steps_without_parsing_errors))
            return super().tape_to_messages(cleaned_tape, steps_description)

    def generate_steps(
        self, agent: Agent, tape: WebTape, llm_stream: LLMStream
    ) -> Generator[Step | PartialStep, None, None]:
        """
        Generates a sequence of steps based on the LLM stream output.

        This method processes the output from a language model stream and converts it into a series of steps.
        It handles the parsing of completions and post-processing of steps.

        Modifications from the original MonoNode:
        - If the last step is an LLMOutputParsingFailureAction, retry the current node
          up to max_retries times. Otherwise, use the default node selection logic.

        Args:
            agent (Any): The agent instance that will execute the steps.
            tape (Tape): The tape object containing the execution context and history.
            llm_stream (LLMStream): The stream of language model outputs to process.

        Yields:
            Union[Step, PartialStep]: Individual steps generated from the LLM stream output.

        Raises:
            FatalError: If no completions are generated from the LLM stream.

        Note:
            - If the node has a next_node defined and the final step is not a StopStep,
              it will yield a SetNextNode step to continue the execution flow.
        """
        # get the last action and the number of repetitions
        previous_actions = [step for step in tape.steps if isinstance(step, Action)]
        last_action = previous_actions[-1] if previous_actions else None
        n_last_actions = len(list(takewhile(lambda x: x == last_action, reversed(previous_actions))))

        new_steps = []
        try:
            cnt = 0
            for event in llm_stream:
                if event.output:
                    cnt += 1
                    assert event.output.content
                    for step in self.parse_completion(event.output.content, llm_stream.prompt.id):
                        step = self.postprocess_step(tape, new_steps, step)
                        # check that the new step is not a repetition of the last action up to max_same_action times
                        if isinstance(step, Action):
                            if step == last_action:
                                n_last_actions += 1
                                if n_last_actions > self.max_same_action:
                                    raise FatalError(f"Max same action reached! {step}")
                            else:
                                n_last_actions = 1
                                last_action = step

                        new_steps.append(step)
                        yield step

                        # if the last step was an LLMOutputParsingFailureAction, retry the current node up to max_retries times
                        if isinstance(step, LLMOutputParsingFailureAction):
                            if self.current_retries < self.max_retries:
                                retry_step = SetNextNode(next_node=self.name)
                                new_steps.append(retry_step)
                                yield retry_step
                                self.current_retries += 1
                                break
                            else:
                                raise FatalError(f"Max retries reached for node {self.name}!")
                        else:
                            self.current_retries = 0
            if not cnt:
                raise FatalError("No completions!")
        except FatalError:
            raise

        if self.next_node and not isinstance(new_steps[-1], StopStep) and not isinstance(new_steps[-1], SetNextNode):
            yield SetNextNode(next_node=self.next_node)

class WebAgent(Agent):
    @classmethod
    def create(cls, llm: LLM, max_iterations: int = 4):
        return super().create(
            llm,
            nodes=[
                WebNode(
                    name="set_goal",
                    guidance=PromptRegistry.start,
                    system_prompt=PromptRegistry.system_prompt,
                    steps_prompt=PromptRegistry.allowed_steps,
                    agent_steps=WebAgentStep,
                ),
                WebNode(
                    name="reflect",
                    guidance=PromptRegistry.reflect,
                    system_prompt=PromptRegistry.system_prompt,
                    steps_prompt=PromptRegistry.allowed_steps,
                    agent_steps=WebAgentStep,
                ),
                WebNode(
                    name="act",
                    guidance=PromptRegistry.act,
                    system_prompt=PromptRegistry.system_prompt,
                    steps_prompt=PromptRegistry.allowed_steps,
                    agent_steps=WebAgentStep,
                    next_node="reflect",
                ),
            ],
            max_iterations=max_iterations,
            store_llm_calls=True,
        )


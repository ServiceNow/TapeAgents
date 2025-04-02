import json
from itertools import takewhile
from typing import Generator

from tapeagents.agent import Agent
from tapeagents.core import (
    Action,
    AgentStep,
    LLMOutputParsingFailureAction,
    PartialStep,
    SetNextNode,
    Step,
    StopStep,
)
from tapeagents.dialog_tape import UserStep
from tapeagents.llms import LLM, LLMStream
from tapeagents.nodes import StandardNode
from tapeagents.tools.simple_browser import PageObservation
from tapeagents.utils import FatalError

from .prompts import PromptRegistry
from .steps import (
    ReasoningThought,
    WebAgentAction,
    WebAgentStep,
    WebTape,
    WebTask,
)


class WebNode(StandardNode):
    max_retries: int = 3  # max number of times to retry the current node if the last step is an LLMOutputParsingFailureAction
    current_retries: int = 0  # current number of retries for the current node
    max_same_action: int = 4  # max number of times to repeat the same action
    max_chars_page_observation: int = 2000  # max number of characters to keep in PageObservation["text"]


    def tape_to_messages(self, tape: WebTape, steps_description: str) -> list[dict]:
        """
        Converts a Tape object and steps description into a list of messages for LLM conversation.

        Modifications from the original StandardNode:
        - If the last n steps are LLMOutputParsingFailureAction, put the guidance before the error steps
          and add a new guidance message that asks to retry.
        - Do not render older LLMOutputParsingFailureAction steps that got solved.
        - Always render WebTask observations as they contain the instructions
        - Truncate PageObservation steps up to `max_chars_page_observation` chars (instead of 100 by default).

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
        messages: list[dict] = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        if steps_description:
            messages.append({"role": "user", "content": steps_description})
        # ignore the last n consecutive parsing error steps for now, will add them later
        last_parsing_error_steps: list[LLMOutputParsingFailureAction] = []
        for step in reversed(tape.steps):
            if isinstance(step, LLMOutputParsingFailureAction):
                last_parsing_error_steps.append(step)
            else:
                break
        # put back the last n parsing error steps in the original order
        last_parsing_error_steps.reverse()
        # remove all the parsing error steps
        cleaned_steps = [step for step in tape.steps if not isinstance(step, LLMOutputParsingFailureAction)]
        # get the positions of all observations to trim the old ones
        page_observation_idx = [i for i, step in enumerate(cleaned_steps) if isinstance(step, PageObservation)]
        for i, step in enumerate(cleaned_steps):
            role = "assistant" if isinstance(step, AgentStep) else "user"
            if isinstance(step, PageObservation):
                if i not in page_observation_idx[-self.trim_obs_except_last_n:]:
                    # skip old page observations
                    continue
                else:
                    # adaptation from step.short_view()
                    view = step.llm_dict()
                    # del view["kind"]  # remove the kind field
                    if len(view["text"]) > self.max_chars_page_observation:
                        view["text"] = view["text"][:self.max_chars_page_observation] + "..."
                    view = json.dumps(view, indent=2, ensure_ascii=False)
                    # view = f"Page Observation: ```json\n{view}\n```"
            elif isinstance(step, WebTask):
                view = f"Task: {step.task}"
            elif isinstance(step, UserStep):
                view = step.content
            # elif isinstance(step, ReasoningThought):
            #     view = step.reasoning
            else:
                view = step.llm_view()
            messages.append({"role": role, "content": view})
        # add the initial guidance message
        if self.guidance:
            messages.append({"role": "user", "content": self.guidance})
        # add the last n parsing error steps and the new guidance message that asks to retry
        for step in last_parsing_error_steps:
            messages.append({"role": "assistant", "content": step.llm_view()})
            messages.append({"role": "user", "content": "You made a mistake. Look at your generation (in 'llm_output') and the error message (in 'error') and please try again."})
        return messages

    def generate_steps(
        self, agent: Agent, tape: WebTape, llm_stream: LLMStream
    ) -> Generator[Step | PartialStep, None, None]:
        """
        Generates a sequence of steps based on the LLM stream output.

        This method processes the output from a language model stream and converts it into a series of steps.
        It handles the parsing of completions and post-processing of steps.

        Modifications from the original StandardNode:
        - If the last step is an LLMOutputParsingFailureAction, retry the current node
          **up to max_retries times**.
        - If the last action is repeated more than **max_same_action times**, stop the task.

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
        for event in llm_stream:
            if not event.output:
                continue
            if event.output.content:
                new_steps += list(self.parse_completion(event.output.content))
            if event.output.tool_calls and self.use_function_calls:
                new_steps += [self.tool_call_to_step(tool_call) for tool_call in event.output.tool_calls]

            for i, step in enumerate(new_steps):
                step = self.postprocess_step(tape, new_steps[:i], step)
                yield step
                # check that the new step is not a repetition of the last action up to max_same_action times
                if isinstance(step, Action):
                    if step == last_action:
                        n_last_actions += 1
                        if n_last_actions > self.max_same_action:
                            raise FatalError(f"Max same action reached! {step}")
                    else:
                        n_last_actions = 1
                        last_action = step
                # if the last step was an LLMOutputParsingFailureAction, retry the current node up to max_retries times
                if isinstance(step, LLMOutputParsingFailureAction):
                    if self.current_retries < self.max_retries:
                        retry_step = SetNextNode(next_node=self.name)
                        yield retry_step
                        self.current_retries += 1
                        new_steps.append(retry_step)
                        break
                    else:
                        raise FatalError(f"Max retries reached for node {self.name}!")
                else:
                    self.current_retries = 0
        if not new_steps:
            raise FatalError("No completions!")
        if (
            self.next_node
            and not isinstance(new_steps[-1], StopStep)
            and not any(isinstance(step, SetNextNode) for step in new_steps)
        ):
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
                    steps=ReasoningThought,
                    trim_obs_except_last_n=3,  # keep the last 3 observations from the tape in prompt messages
                    max_chars_page_observation=3000,  # keep up to 3000 chars in PageObservation steps
                ),
                WebNode(
                    name="reflect",
                    guidance=PromptRegistry.reflect,
                    system_prompt=PromptRegistry.system_prompt,
                    steps_prompt=PromptRegistry.allowed_steps,
                    steps=WebAgentStep,
                    trim_obs_except_last_n=3,  # keep the last 3 observations from the tape in prompt messages
                    max_chars_page_observation=3000,  # keep up to 3000 chars in PageObservation steps
                ),
                WebNode(
                    name="act",
                    guidance=PromptRegistry.act,
                    system_prompt=PromptRegistry.system_prompt,
                    steps_prompt=PromptRegistry.allowed_steps,
                    steps=WebAgentAction,
                    trim_obs_except_last_n=3,  # keep the last 3 observations from the tape in prompt messages
                    max_chars_page_observation=3000,  # keep up to 3000 chars in PageObservation steps
                    next_node="reflect",
                ),
            ],
            max_iterations=max_iterations,
            store_llm_calls=True,
        )


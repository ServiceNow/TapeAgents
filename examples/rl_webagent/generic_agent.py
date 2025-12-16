"""Generic Web Agent implementation using prompts from AgentLab."""

from __future__ import annotations

import ast
import json
import re
from itertools import takewhile
from typing import Any, Generator

from examples.rl_webagent.utils import StepRepeatMonitor
from tapeagents.agent import Agent
from tapeagents.core import (
    Action,
    LLMOutputParsingFailureAction,
    PartialStep,
    Prompt,
    SetNextNode,
    Step,
    StopStep,
)
from tapeagents.llms import LLM, LLMStream
from tapeagents.steps import ReasoningThought
from tapeagents.tools.browser import (
    ClickBIDAction,
    ClickCoordinatesAction,
    GoBackAction,
    GoForwardAction,
    HoverAction,
    InputTextAction,
    PageObservation,
    PressAction,
    SelectOptionAction,
)
from tapeagents.utils import FatalError

from .agent import WebNode
from .prompts_generic import GenericPromptRegistry
from .steps import FinalAnswerAction, SendMessageToUserAction, WebAgentAction, WebTape, WebTask

_THINK_PATTERN = re.compile(r"<think>(.*?)</think>", re.IGNORECASE | re.DOTALL)
_ACTION_PATTERN = re.compile(r"<action>(.*?)</action>", re.IGNORECASE | re.DOTALL)


class GenericWebNode(WebNode):
    """Web node that formats prompts using the Generic Agent template."""

    include_examples: bool = True
    include_think_in_history: bool = True

    def make_prompt(self, agent: Any, tape: WebTape) -> Prompt:
        """Create the LLM prompt following the AgentLab generic structure."""

        task_step = next((s for s in tape.steps if isinstance(s, WebTask)), None)
        goal = task_step.task if task_step else "Complete the task"

        obs_steps = [s for s in tape.steps if isinstance(s, PageObservation)]
        if obs_steps:
            latest_obs = obs_steps[-1]
            html_content = getattr(latest_obs, "text", "") or ""
            error_content = getattr(latest_obs, "error", "") or ""
            
            # If latest observation has no HTML, try to get HTML from previous observation
            if not html_content and len(obs_steps) > 1:
                for prev_obs in reversed(obs_steps[:-1]):
                    prev_html = getattr(prev_obs, "text", "") or ""
                    if prev_html:
                        html_content = prev_html
                        break
            
            if html_content and self.max_chars_page_observation and len(html_content) > self.max_chars_page_observation:
                html_content = html_content[: self.max_chars_page_observation] + "..."
            
            if html_content:
                observation_text = f"## HTML\n{html_content}"
                if error_content:
                    observation_text += f"\n\n## Error from previous action:\n{error_content}"
            elif error_content:
                # No HTML but there's an error - show the error
                observation_text = f"## Error from previous action:\n{error_content}\n\n(No HTML content available)"
            elif hasattr(latest_obs, "short_view"):
                observation_text = latest_obs.short_view(max_chars=self.max_chars_page_observation)
            else:
                observation_text = latest_obs.llm_view()
        else:
            observation_text = "No observation available."

        history_text = self._render_history_blocks(tape)

        action_space_text = (
            f"{GenericPromptRegistry.action_set_info}\n\n"
            f"{GenericPromptRegistry.action_space_description}"
        )

        template = GenericPromptRegistry.complete_prompt_template
        if not self.include_examples:
            template = template.split("# Abstract Example", 1)[0]

        user_content = template.format(
            goal=goal,
            observation=observation_text,
            history=history_text,
            action_space=action_space_text,
            hints=GenericPromptRegistry.hints,
        )

        system_content = self.system_prompt or GenericPromptRegistry.system_prompt

        return Prompt(
            messages=[
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_content},
            ]
        )

    def _render_history_blocks(self, tape: WebTape) -> str:
        history_lines: list[str] = []
        step_index = 0
        pending_think: str | None = None
        has_entries = False

        for step in tape.steps:
            if isinstance(step, ReasoningThought):
                pending_think = step.reasoning.strip()
                continue

            if isinstance(step, LLMOutputParsingFailureAction):
                # Add parsing error as an action step without think block
                history_lines.append(f"# Step {step_index}")
                history_lines.append("<action>")
                history_lines.append(f"Error: {step.error}")
                history_lines.append("</action>")
                history_lines.append("")
                step_index += 1
                pending_think = None
                has_entries = True
                continue

            if isinstance(step, Action):
                action_text = self._format_action_history_entry(step)
                history_lines.append(f"# Step {step_index}")
                if self.include_think_in_history:
                    think_text = pending_think or "No reasoning provided."
                    history_lines.append("<think>")
                    history_lines.append(think_text)
                    history_lines.append("</think>")
                history_lines.append("<action>")
                history_lines.append(action_text)
                history_lines.append("</action>")
                history_lines.append("")
                step_index += 1
                pending_think = None
                has_entries = True

        if not has_entries:
            return "# History of interaction with the task:\nNo previous actions yet.\n"

        return "\n".join(history_lines).strip() + "\n"

    def generate_steps(
        self, agent: Agent, tape: WebTape, llm_stream: LLMStream
    ) -> Generator[Step | PartialStep, None, None]:
        """Generate steps using think/action parsing for generic outputs.
        
        When vLLM reasoning parser is enabled, the reasoning is returned in the
        output.reasoning field, and the action is in output.content.
        Otherwise, we parse <think> and <action> blocks from output.content.
        """

        previous_actions = [step for step in tape.steps if isinstance(step, Action)]
        last_action = previous_actions[-1] if previous_actions else None
        n_last_actions = len(list(takewhile(lambda x: x == last_action, reversed(previous_actions))))
        action_monitor = StepRepeatMonitor(
            last_step=last_action, repeat_count=n_last_actions, max_repeats=self.max_same_action
        )

        new_steps: list[Step] = []
        for event in llm_stream:
            if not event.output:
                continue
            
            # Check if vLLM reasoning parser provided reasoning separately
            reasoning = getattr(event.output, "reasoning", None)
            content = event.output.content
            
            if hasattr(agent, "llm") and hasattr(agent.llm, "tokenizer") and agent.llm.tokenizer:
                eos_token = agent.llm.tokenizer.eos_token
                if eos_token:
                    if content and content.endswith(eos_token):
                        content = content[: -len(eos_token)]
                    if reasoning and reasoning.endswith(eos_token):
                        reasoning = reasoning[: -len(eos_token)]
            
            if reasoning is not None:
                # vLLM reasoning parser mode: reasoning is separate from content
                if reasoning:
                    new_steps.append(ReasoningThought(reasoning=reasoning.strip()))
                
                if content:
                    # Content contains the action (without <action> tags in vLLM mode)
                    # Try to parse as action directly
                    try:
                        action_step = self._parse_action_string(content.strip())
                        if action_step:
                            new_steps.append(action_step)
                        else:
                            # Empty action content
                            new_steps.append(
                                LLMOutputParsingFailureAction(
                                    error="Empty action content from vLLM reasoning output",
                                    llm_output=content,
                                )
                            )
                    except Exception as exc:
                        # Maybe content still has <action> tags
                        action_match = _ACTION_PATTERN.search(content)
                        if action_match:
                            try:
                                action_step = self._parse_action_string(action_match.group(1))
                                if action_step:
                                    new_steps.append(action_step)
                            except Exception as inner_exc:
                                new_steps.append(
                                    LLMOutputParsingFailureAction(
                                        error=f"Failed to parse action: {inner_exc}",
                                        llm_output=content,
                                    )
                                )
                        else:
                            new_steps.append(
                                LLMOutputParsingFailureAction(
                                    error=f"Failed to parse action from vLLM output: {exc}",
                                    llm_output=content,
                                )
                            )
                elif not reasoning:
                    new_steps.append(
                        LLMOutputParsingFailureAction(
                            error="Empty output from vLLM (no reasoning or content)",
                            llm_output="",
                        )
                    )
            elif content:
                # Fallback: parse <think> and <action> blocks from content
                parsed_steps = self._parse_think_action_completion(content)
                if parsed_steps:
                    new_steps.extend(parsed_steps)
                else:
                    new_steps.append(
                        LLMOutputParsingFailureAction(
                            error="Missing <think>/<action> blocks in LLM output",
                            llm_output=content,
                        )
                    )

            if event.output.tool_calls and self.use_function_calls:
                new_steps.extend(self.tool_call_to_step(agent, tool_call) for tool_call in event.output.tool_calls)

            for index, step in enumerate(new_steps):
                processed_step = self.postprocess_step(tape, new_steps[:index], step)
                yield processed_step

                if isinstance(processed_step, Action) and action_monitor.should_stop(processed_step):
                    raise FatalError(f"Max same action reached! {processed_step}")

                if isinstance(processed_step, LLMOutputParsingFailureAction):
                    if self.current_retries < self.max_retries:
                        retry_step = SetNextNode(next_node=self.name)
                        yield retry_step
                        self.current_retries += 1
                        new_steps.append(retry_step)
                        break

                    raise FatalError(
                        f"Max retries reached for node {self.name}, parsing error: {processed_step.llm_view()}!"
                    )

                self.current_retries = 0

        if not new_steps:
            raise FatalError("No completions!")

        if (
            self.next_node
            and not isinstance(new_steps[-1], StopStep)
            and not any(isinstance(step, SetNextNode) for step in new_steps)
        ):
            yield SetNextNode(next_node=self.next_node)

    def _format_action_history_entry(self, action: Action) -> str:
        call_repr = self._action_to_call(action)
        return call_repr

    def _action_to_call(self, action: Action) -> str:
        if isinstance(action, ClickBIDAction):
            args = [repr(action.bid)]
            if action.button != "left":
                args.append(f"button={repr(action.button)}")
            if action.modifiers:
                args.append(f"modifiers={repr(list(action.modifiers))}")
            return f"click({', '.join(args)})"

        if isinstance(action, InputTextAction):
            args = [repr(action.bid), repr(action.text)]
            return f"fill({', '.join(args)})"

        if isinstance(action, SelectOptionAction):
            option_arg = repr(action.option)
            args = [repr(action.bid), option_arg]
            if action.element_description:
                args.append(f"element_description={repr(action.element_description)}")
            return f"select_option({', '.join(args)})"

        if isinstance(action, HoverAction):
            return f"hover({repr(action.bid)})"

        if isinstance(action, PressAction):
            return f"press({repr(action.bid)}, {repr(action.key_comb)})"

        if isinstance(action, ClickCoordinatesAction):
            args = [self._format_number(action.x), self._format_number(action.y)]
            if action.button != "left":
                args.append(f"button={repr(action.button)}")
            return f"click_coordinates({', '.join(args)})"

        if isinstance(action, GoBackAction):
            return "go_back()"

        if isinstance(action, GoForwardAction):
            return "go_forward()"

        if isinstance(action, FinalAnswerAction):
            return f"final_answer({repr(action.text)})"

        if isinstance(action, SendMessageToUserAction):
            return f"send_msg_to_user({json.dumps(action.text)})"

        return action.llm_view()

    @staticmethod
    def _format_number(value: float) -> str:
        if value.is_integer():
            return str(int(value))
        return repr(value)

    def _parse_think_action_completion(self, llm_output: str) -> list[Step]:
        """Parse <think> and <action> blocks into structured steps."""

        think_match = _THINK_PATTERN.search(llm_output)
        action_match = _ACTION_PATTERN.search(llm_output)

        if not think_match and not action_match:
            return []

        steps: list[Step] = []

        if think_match:
            reasoning_text = think_match.group(1).strip()
            if reasoning_text:
                steps.append(ReasoningThought(reasoning=reasoning_text))

        if action_match:
            action_block = action_match.group(1)
            try:
                action_step = self._parse_action_string(action_block)
            except Exception as exc:
                steps.append(
                    LLMOutputParsingFailureAction(
                        error=f"Failed to parse <action> block: {exc}",
                        llm_output=llm_output,
                    )
                )
            else:
                if action_step is not None:
                    steps.append(action_step)

        return steps

    def _parse_action_string(self, action_block: str) -> Step | None:
        """Convert the text inside <action>...</action> into a Step instance."""

        cleaned_lines = []
        for line in action_block.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            if "#" in stripped:
                stripped = stripped.split("#", 1)[0].strip()
            if stripped:
                cleaned_lines.append(stripped)

        action_expr = " ".join(cleaned_lines)
        if not action_expr:
            return None

        try:
            parsed = ast.parse(action_expr, mode="eval")
        except SyntaxError as exc:
            raise ValueError(f"Invalid action expression: {exc.msg}") from exc

        if not isinstance(parsed.body, ast.Call):
            raise ValueError("Action expression must be a function call")

        func_node = parsed.body.func
        if not isinstance(func_node, ast.Name):
            raise ValueError("Unsupported action function expression")

        func_name = func_node.id.lower()
        args = [ast.literal_eval(arg) for arg in parsed.body.args]
        kwargs = {kw.arg: ast.literal_eval(kw.value) for kw in parsed.body.keywords if kw.arg}

        if func_name in {"click", "dblclick"}:
            bid = str(args[0]) if args else str(kwargs.get("bid", ""))
            if not bid:
                raise ValueError("click action missing bid")
            button = kwargs.get("button", "left")
            modifiers = kwargs.get("modifiers", [])
            if not isinstance(modifiers, list):
                raise ValueError("click modifiers must be a list")
            return ClickBIDAction(bid=bid, button=button, modifiers=modifiers)

        if func_name in {"fill", "input_text", "type_text"}:
            bid = str(args[0]) if args else str(kwargs.get("bid", ""))
            if not bid:
                raise ValueError("input action missing bid")
            if len(args) > 1:
                text = args[1]
            else:
                text = kwargs.get("value") or kwargs.get("text")
            if text is None:
                raise ValueError("input action missing value/text")
            return InputTextAction(bid=bid, text=str(text))

        if func_name == "select_option":
            bid = str(args[0]) if args else str(kwargs.get("bid", ""))
            if not bid:
                raise ValueError("select_option action missing bid")
            if len(args) > 1:
                option = args[1]
            else:
                option = kwargs.get("options") or kwargs.get("option")
            if option is None:
                raise ValueError("select_option action missing option")
            option_value = option[0] if isinstance(option, list) else option
            element_description = kwargs.get("element_description", "")
            return SelectOptionAction(
                bid=bid,
                element_description=str(element_description),
                option=str(option_value),
            )

        if func_name == "hover":
            bid = str(args[0]) if args else str(kwargs.get("bid", ""))
            if not bid:
                raise ValueError("hover action missing bid")
            return HoverAction(bid=bid)

        if func_name == "press":
            bid = str(args[0]) if args else str(kwargs.get("bid", ""))
            key_comb = args[1] if len(args) > 1 else kwargs.get("key_comb")
            if not bid or key_comb is None:
                raise ValueError("press action requires bid and key combination")
            return PressAction(bid=bid, key_comb=str(key_comb))

        if func_name == "click_coordinates":
            if len(args) < 2:
                raise ValueError("click_coordinates requires x and y arguments")
            x_val = float(args[0])
            y_val = float(args[1])
            button = kwargs.get("button", "left")
            return ClickCoordinatesAction(x=x_val, y=y_val, button=button)

        if func_name == "go_back":
            return GoBackAction()

        if func_name == "go_forward":
            return GoForwardAction()

        if func_name == "final_answer":
            if args:
                text = args[0]
            else:
                text = kwargs.get("text", "")
            return FinalAnswerAction(text=str(text))

        if func_name == "send_msg_to_user":
            if args:
                text = args[0]
            else:
                text = kwargs.get("text") or kwargs.get("message")
            if text is None:
                raise ValueError("send_msg_to_user action missing text")
            return SendMessageToUserAction(text=str(text))

        raise ValueError(f"Unsupported action '{func_name}'")


class GenericWebAgent(Agent):
    """Generic Web Agent that uses AgentLab-style prompts with thinking and action structure."""

    @classmethod
    def create(cls, llm: LLM, max_iterations: int = 15, use_examples: bool = True):
        """Create a Generic Web Agent with a single think-and-act node."""

        return super().create(
            llm,
            nodes=[
                GenericWebNode(
                    name="think_and_act",
                    guidance="",
                    system_prompt=GenericPromptRegistry.system_prompt,
                    steps_prompt="",
                    steps=WebAgentAction,
                    trim_obs_except_last_n=3,
                    max_chars_page_observation=10000,
                    include_examples=use_examples,
                    next_node="think_and_act",
                ),
            ],
            max_iterations=max_iterations,
            store_llm_calls=True,
        )

import json
from typing import Any, Type

import yaml
from pydantic import BaseModel

from tapeagents.core import Episode, Observation, Step, Tape
from tapeagents.dialog_tape import UserStep
from tapeagents.llms import LLMCall


class BasicRenderer:
    """A basic renderer for displaying tapes in HTML format.

    This class provides functionality to render tapes and LLM calls in a structured HTML format
    with customizable styling and filtering options.

    Attributes:
        metadata_header (str): HTML header for metadata section
        context_header (str): HTML header for context section
        steps_header (str): HTML header for steps section
        agent_tape_header (str): HTML header for agent tape section
        user_tape_header (str): HTML header for user tapes section
        annotator_tape_header (str): HTML header for annotator tapes section

    Args:
        filter_steps (Optional[tuple[Type, ...]]): Types of steps to include in rendering. If None, all steps are rendered.
        render_llm_calls (bool): Whether to render LLM calls. Defaults to True.
        render_agent_node (bool): Whether to render agent node information. Defaults to False.

    Example:
        ```python
        renderer = BasicRenderer(render_llm_calls=True)
        html_output = renderer.render_tape(tape)
        ```

    The renderer supports:

    - Tape rendering with metadata, context, and steps
    - Episode rendering with user, agent, and annotator columns
    - LLM call rendering with prompts and outputs
    - Custom styling through CSS
    - Step filtering
    - Collapsible sections using HTML details tags
    """

    metadata_header = "<h3 style='margin: 2px'>Metadata</h3>"
    context_header = "<h3 style='margin: 2px'>Context</h3>"
    steps_header = "<h3 style='margin: 2px'>Steps</h3>"
    agent_tape_header = "<h1> Agent Tape </h1>"
    user_tape_header = "<h1> User Tapes </h1>"
    annotator_tape_header = "<h1> Annotator Tapes </h1>"

    def __init__(
        self,
        filter_steps: tuple[Type, ...] | None = None,
        render_llm_calls: bool = True,
        render_agent_node: bool = False,
    ):
        self.filter_steps = filter_steps
        self.render_llm_calls = render_llm_calls
        self.render_agent_node = render_agent_node

    @property
    def style(self) -> str:
        return (
            "<style>"
            ".basic-renderer-box { margin: 4px; padding: 2px; padding-left: 6px; background: lavender; white-space: pre-wrap; color: black;}"
            ".basic-prompt-box { margin: 4px; padding: 2px; padding-left: 6px; background: lavender; color: black;}"
            ".episode-row { display: flex; align-items: end; }"
            ".agent-column { width: 50%; }"
            ".user-column { width: 25%; }"
            ".annotator-column { width: 25%; }"
            ".inner-tape-container { display: flex }"
            ".inner-tape-indent { width: 10%; }"
            ".inner-tape { width: 90%; }"
            ".agent_node { text-align: center; position: relative; margin: 12px 0; }"
            ".agent_node hr { border: 1px solid #e1e1e1; margin: 0 !important;}"
            ".agent_node span { position: absolute; right: 0; top: -0.6em; background: white; color: grey !important; padding: 0 10px; }"
            "</style>"
        )

    def render_as_box(self, data: Any):
        if isinstance(data, dict):
            str_ = yaml.dump(data, indent=2)
        else:
            str_ = str(data)
        return f"<div class='basic-renderer-box'>{str_}</div>"

    def render_metadata(self, tape: Tape):
        return f"<details> <summary>id: {tape.metadata.id}</summary> {self.render_as_box(tape.metadata.model_dump())} </details>"

    def render_context(self, tape: Tape):
        if isinstance(tape.context, Tape):
            summary = f"Tape of {len(tape.context)} steps." f"<div>ID: {tape.context.metadata.id}</div>"
            return (
                "<div class=inner-tape-container>"
                "<div class=inner-tape-indent> </div>"
                "<div class=inner-tape> <details>"
                f"<summary>{summary}</b></summary>"
                f"{self.render_tape(tape.context)}</details>"
                "</div></div>"
            )
        else:
            context_str = tape.context.model_dump() if isinstance(tape.context, BaseModel) else tape.context
            return self.render_as_box(context_str)

    def render_step(self, step: Step, index: int, **kwargs) -> str:
        """
        Renders a single step in the process.

        Args:
            step (Step): The step object containing the data to be rendered
            index (int): The index of the current step
            **kwargs (dict, optional): Additional keyword arguments for rendering customization

        Returns:
            str: The rendered step as a formatted string in box format

        Note:
            The step is first converted to a dictionary using model_dump() before rendering.
            The rendering is done using the render_as_box method.
        """
        step_dict = step.model_dump()
        return self.render_as_box(step_dict)

    def render_steps(self, tape: Tape, llm_calls: dict[str, LLMCall] = {}) -> str:
        """
        Renders a sequence of steps from a tape into an HTML string representation.

        This method processes each step in the tape and generates HTML chunks based on various rendering options.
        It can show agent nodes, LLM calls, and individual steps based on configuration.

        Args:
            tape (Tape): The tape object containing the sequence of steps to render
            llm_calls (dict[str, LLMCall], optional): Dictionary mapping prompt IDs to LLM calls. Defaults to {}.

        Returns:
            str: A concatenated HTML string containing all rendered chunks

        Notes:
            - If filter_steps is set, only steps matching those types will be rendered
            - Agent nodes are rendered as dividers with agent/node names if render_agent_node is True
            - LLM calls are rendered for each unique prompt_id if render_llm_calls is True
            - Steps from UserStep and Observation are treated as "Environment" agent
        """
        chunks = []
        last_prompt_id = None
        last_agent_node = None
        for index, step in enumerate(tape):
            if self.filter_steps and not isinstance(step, self.filter_steps):
                continue
            if self.render_agent_node:
                if isinstance(step, UserStep) or isinstance(step, Observation):
                    agent = "Environment"
                    node = ""
                else:
                    agent = step.metadata.agent.split("/")[-1]
                    node = step.metadata.node
                agent_node = agent + (f".{node}" if node else "")
                if agent_node != last_agent_node:
                    chunks.append(f"""<div class="agent_node"><hr><span>{agent_node}</span></div>""")
                    last_agent_node = agent_node
            if self.render_llm_calls:
                if step.metadata.prompt_id != last_prompt_id:
                    llm_call = llm_calls.get(step.metadata.prompt_id)
                    if llm_call:
                        chunks.append(self.render_llm_call(llm_call))
                    last_prompt_id = step.metadata.prompt_id
            chunks.append(self.render_step(step, index))
        return "".join(chunks)

    def render_tape(self, tape: Tape, llm_calls: dict[str, LLMCall] = {}) -> str:
        """
        Render a tape object into HTML representation.

        Args:
            tape (Tape): The tape object to render.
            llm_calls (dict[str, LLMCall], optional): Dictionary of LLM calls associated with the tape. Defaults to {}.

        Returns:
            str: HTML representation of the tape including metadata, context (if present), and steps.

        The rendered HTML includes:

        - Metadata section with tape metadata
        - Context section (if tape.context exists)
        - Steps section with tape execution steps
        """
        metadata_html = self.render_metadata(tape)
        context_html = self.render_context(tape)
        steps_html = self.render_steps(tape, llm_calls)
        return (
            f"{self.metadata_header}{metadata_html}"
            + (f"{self.context_header}{context_html}" if tape.context is not None else "")
            + f"{self.steps_header}{steps_html}"
        )

    def render_episode(self, episode: Episode) -> str:
        """Renders an episode into HTML format.

        Takes an Episode object and converts it into an HTML string representation with three columns:
        user, agent, and annotator. The rendering includes headers, context, and sequential steps
        organized in rows.

        Args:
            episode (Episode): Episode object containing the interaction sequence to be rendered

        Returns:
            str: HTML string representation of the episode with formatted columns and rows
        """
        chunks = []

        def wrap_agent(html: str) -> str:
            return f"<div class='agent-column'>{html}</div>"

        def wrap_user(html: str) -> str:
            return f"<div class='user-column'>{html}</div>"

        def wrap_annotator(html: str) -> str:
            return f"<div class='annotator-column'>{html}</div>"

        def row(user: str, agent: str, annotator: str):
            return f"<div class='episode-row'>{wrap_user(user)}{wrap_agent(agent)}{wrap_annotator(annotator)}</div>"

        chunks.append(row(self.user_tape_header, self.agent_tape_header, self.annotator_tape_header))
        chunks.append(row("", self.context_header, ""))
        chunks.append(row("", self.render_context(episode.tape), ""))
        chunks.append(row("", self.steps_header, ""))
        for index, (user_tape, step, annotator_tapes) in enumerate(episode.group_by_step()):
            if user_tape:
                user_html = f"{self.steps_header}{self.render_steps(user_tape)}"
            else:
                user_html = ""
            agent_html = self.render_step(step, index)
            annotations_html = "".join([f"{self.steps_header}{self.render_steps(tape)}" for tape in annotator_tapes])
            chunks.append(row(user_html, agent_html, annotations_html))
        return "".join(chunks)

    def render_llm_call(self, llm_call: LLMCall | None) -> str:
        """Renders an LLM call into HTML format.

        This method generates HTML representation of an LLM call, including both the prompt
        and output (if available). The HTML includes expandable details sections for both
        the prompt and response.

        Args:
            llm_call (Union[LLMCall, None]): An LLM call object containing prompt and output information.
                If None, returns an empty string.

        Returns:
            str: HTML string representation of the LLM call. The HTML contains:

            - Prompt section with:
                - Summary showing token/character count and cache status
                - Expandable details with prompt messages
            - Output section (if output exists) with:
                - Summary showing token count
                - Expandable details with LLM response

        The rendered HTML uses collapsible details elements and basic styling for
        readability, with a light yellow background color.
        """
        if llm_call is None:
            return ""
        if llm_call.prompt.tools:
            prompt_messages = [f"tool_schemas: {json.dumps(llm_call.prompt.tools, indent=2)}"]
        else:
            prompt_messages = []
        for m in llm_call.prompt.messages:
            role = f"{m['role']} ({m['name']})" if "name" in m else m["role"]
            prompt_messages.append(f"{role}: {m['content'] if 'content' in m else m['tool_calls']}")
        prompt_text = "\n--\n".join(prompt_messages)
        prompt_length_str = (
            f"{llm_call.prompt_length_tokens} tokens"
            if llm_call.prompt_length_tokens
            else f"{len(prompt_text)} characters"
        )
        label = f"Prompt {prompt_length_str} {', cached' if llm_call.cached else ''}"
        html = f"""<div class='basic-prompt-box' style='background-color:#ffffba; margin: 0 4px;'>
        <details>
            <summary>{label}</summary>
            <pre style='font-size: 12px; white-space: pre-wrap;word-wrap: break-word;'>{prompt_text.strip()}</pre>
        </details>
        </div>"""
        if llm_call.output:
            html += f"""<div class='basic-prompt-box' style='background-color:#ffffba; margin: 0 4px;'>
                <details>
                    <summary>LLM Output {llm_call.output_length_tokens} tokens</summary>
                    <pre style='font-size: 12px; white-space: pre-wrap; word-wrap: break-word;'>{llm_call.output}</pre>
                </details>
                </div>"""
        return html

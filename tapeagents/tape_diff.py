import os
from itertools import zip_longest
from typing import Any, Callable

import gradio as gr

from tapeagents.rendering import BasicRenderer
from tapeagents.utils import diff_strings


class TapeDiffGUI:
    """
    GUI for comparing two set of tapes from different runs.
    Assuming that the tapes are stored in JSON files with the same structure and produced from the same tasks.
    So the tape N from the first file is compared with the tape N from the second file.
    """

    def __init__(
        self,
        renderer: BasicRenderer,
        loader_fn: Callable[[str], tuple[list, dict, str]],
        tape_name_fn: Callable[[int, Any, Any], str],
        fname: str,
        fname2: str,
        file_extension: str = ".json",
    ):
        self.renderer = renderer
        self.dirname = os.path.dirname(fname)
        self.dirname2 = os.path.dirname(fname2)
        self.fname = os.path.basename(fname)
        self.fname2 = os.path.basename(fname2)
        self.loader = loader_fn
        self.tape_name = tape_name_fn
        self.file_extension = file_extension

    def on_load(self):
        files = sorted([f for f in os.listdir(self.dirname) if f.endswith(self.file_extension)])
        files2 = sorted([f for f in os.listdir(self.dirname2) if f.endswith(self.file_extension)])
        file_selector = gr.Dropdown(files, label="File A", value=self.fname)  # type: ignore
        file_selector2 = gr.Dropdown(files2, label="File B", value=self.fname2)  # type: ignore
        return file_selector, file_selector2

    def update(self, fname: str, fname2: str, n: int, first_time: bool = False):
        tapes, prompts, header = self.loader(os.path.join(self.dirname, fname))
        tapes2, prompts2, header2 = self.loader(os.path.join(self.dirname2, fname2))
        tape_names = []

        # prepare list of tape names
        for i, tape in enumerate(tapes):
            name = self.tape_name(i, tape, tapes2[i] if i < len(tapes2) else None)
            tape_names.append((name, i))
        tape_names2 = []
        for i, tape in enumerate(tapes2):
            name = self.tape_name(i, tape, tapes[i] if i < len(tapes) else None)
            tape_names2.append((name, i))
        if not first_time:
            tape_names = gr.Dropdown(tape_names, label="Tape A", value=n)
            tape_names2 = gr.Dropdown(tape_names2, label="Tape B", value=n)

        # render pair of tapes
        style = "<style>.basic-renderer-box { width: 600px !important; }</style>"
        html = f"""{self.renderer.style}{style}<table class="diff" cellspacing="0" cellpadding="0"><tr class="diff">
            <td class="diff"><h2 style="padding-left: 2em;">{header}</h2></td>
            <td class="diff"><h2 style="padding-left: 2em;">{header2}</h2></td></tr>"""
        for i, (step_dict, step_dict2) in enumerate(zip_longest(tapes[n]["steps"], tapes2[n]["steps"])):
            prompt_text = ""
            diff_class = "diff"
            if step_dict is None:
                step = ""
            else:
                step = self.renderer.render_step(step_dict, i, folded=False)
                prompt_id = step_dict.pop("prompt_id", None)
                if prompt_id in prompts:
                    prompt_text = self.renderer.render_llm_call(prompts[prompt_id])
                    step = prompt_text + step
            if step_dict2 is None:
                step2 = ""
            else:
                step2 = self.renderer.render_step(step_dict2, i, folded=False)
                prompt_id2 = step_dict2.pop("prompt_id", None)
                if step_dict and step_dict != step_dict2:
                    diff_class = "diff_highlight"
                    if step_dict["kind"] == step_dict2["kind"]:
                        # highlight differences in step B
                        step2 = diff_strings(
                            self.renderer.render_step(step_dict, i, folded=False), step2, use_html=True, by_words=True
                        )
                if prompt_id2 in prompts2:
                    prompt_text2 = self.renderer.render_llm_call(prompts2[prompt_id2])
                    if prompt_text and prompt_text != prompt_text2:
                        # highlight differences in prompt B
                        prompt_text2 = diff_strings(prompt_text, prompt_text2, use_html=True, by_words=True)
                    step2 = prompt_text2 + step2
            html += f'<tr class="diff"><td class="diff">{step}</td><td class="{diff_class}">{step2}</td></tr>'
        html += "</table>"

        return html, tape_names, tape_names2

    def launch(self, server_name: str = "0.0.0.0"):
        tape = 0
        with gr.Blocks() as blocks:
            html, tape_names, tape_names2 = self.update(self.fname, self.fname2, tape, first_time=True)
            with gr.Row():
                with gr.Column(scale=4):
                    file_selector = gr.Dropdown([], label="File A")
                    selector = gr.Dropdown(tape_names, label="Tape A", value=tape)  # type: ignore
                with gr.Column(scale=4):
                    file_selector2 = gr.Dropdown([], label="File B")
                    selector2 = gr.Dropdown(tape_names2, label="Tape B", value=tape)  # type: ignore
            with gr.Row():
                tape_view = gr.HTML(html)

            selector.change(
                fn=self.update,
                inputs=[file_selector, file_selector2, selector],
                outputs=[tape_view, selector, selector2],
            )
            selector2.change(
                fn=self.update,
                inputs=[file_selector, file_selector2, selector2],
                outputs=[tape_view, selector, selector2],
            )
            file_selector.change(
                fn=self.update,
                inputs=[file_selector, file_selector2, selector],
                outputs=[tape_view, selector, selector2],
            )
            file_selector2.change(
                fn=self.update,
                inputs=[file_selector, file_selector2, selector],
                outputs=[tape_view, selector, selector2],
            )
            blocks.load(self.on_load, inputs=None, outputs=[file_selector, file_selector2])
        blocks.launch(server_name=server_name)

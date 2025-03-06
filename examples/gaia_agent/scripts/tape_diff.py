import os
import sys
from itertools import zip_longest

import gradio as gr

from tapeagents.core import LLMCall
from tapeagents.io import load_tapes
from tapeagents.renderers.basic import BasicRenderer
from tapeagents.utils import diff_strings

from ..eval import calculate_accuracy, tape_correct
from ..steps import GaiaTape


class TapeDiffGUI:
    """
    GUI for comparing two set of tapes from different runs.
    Assuming that the tapes are stored in JSON files with the same structure and produced from the same tasks.
    So the tape N from the first file is compared with the tape N from the second file.
    """

    def __init__(
        self,
        renderer: BasicRenderer,
        fname: str,
        fname2: str,
        file_extension: str = ".json",
    ):
        self.renderer = renderer
        self.dirname = os.path.dirname(fname)
        self.dirname2 = os.path.dirname(fname2)
        self.fname = os.path.basename(fname)
        self.fname2 = os.path.basename(fname2)
        self.file_extension = file_extension

    def load_tapes(self, fname: str) -> tuple[list[GaiaTape], dict[str, LLMCall], str]:
        tapes: list[GaiaTape] = load_tapes(GaiaTape, fname)  # type: ignore
        llm_calls = {}  # TODO: load prompts
        acc, n_solved = calculate_accuracy(tapes)
        header = f"Accuracy: {acc:.2f}%, {n_solved} out of {len(tapes)}"
        return tapes, llm_calls, header

    def tape_name(self, n: int, tape: GaiaTape, tape2: GaiaTape | None) -> str:
        correct = tape_correct(tape)
        task = tape.metadata.task
        mark = "+" if correct else "-"
        if tape2:
            mark += "+" if tape_correct(tape2) else "-"
        if task["file_name"]:
            mark += "[f]"
        return f"{mark}L{task['Level']}:{n+1}: {task['Question'][:32]}"

    def on_load(self):
        files = sorted([f for f in os.listdir(self.dirname) if f.endswith(self.file_extension)])
        files2 = sorted([f for f in os.listdir(self.dirname2) if f.endswith(self.file_extension)])
        file_selector = gr.Dropdown(files, label="File A", value=self.fname)  # type: ignore
        file_selector2 = gr.Dropdown(files2, label="File B", value=self.fname2)  # type: ignore
        return file_selector, file_selector2

    def update(self, fname: str, fname2: str, n: int, first_time: bool = False):
        tapes, llm_calls, header = self.load_tapes(os.path.join(self.dirname, fname))
        tapes2, llm_calls2, header2 = self.load_tapes(os.path.join(self.dirname2, fname2))
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
        for i, (step, step2) in enumerate(zip_longest(tapes[n].steps, tapes2[n].steps)):
            prompt_text = ""
            diff_class = "diff"
            if step is None:
                step_str = ""
            else:
                step_str = self.renderer.render_step(step, i, folded=False)
                prompt_id = step.metadata.prompt_id
                if prompt_id and prompt_id in llm_calls:
                    prompt_text = self.renderer.render_llm_call(llm_calls[prompt_id])
                    step_str = prompt_text + step_str
            if step2 is None:
                step2_str = ""
            else:
                step2_str = self.renderer.render_step(step2, i, folded=False)
                prompt_id2 = step2.metadata.prompt_id
                if step and step != step2:
                    diff_class = "diff_highlight"
                    if step.kind == step2.kind:
                        # highlight differences in step B
                        step2_str = diff_strings(
                            self.renderer.render_step(step, i, folded=False), step2_str, use_html=True, by_words=True
                        )
                if prompt_id2 in llm_calls2:
                    prompt_text2 = self.renderer.render_llm_call(llm_calls2[prompt_id2])
                    if prompt_text and prompt_text != prompt_text2:
                        # highlight differences in prompt B
                        prompt_text2 = diff_strings(prompt_text, prompt_text2, use_html=True, by_words=True)
                    step2_str = prompt_text2 + step2_str
            html += f'<tr class="diff"><td class="diff">{step_str}</td><td class="{diff_class}">{step2_str}</td></tr>'
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


if __name__ == "__main__":
    assert len(sys.argv) == 3, "Usage: python -m scripts.tape_diff <fname1> <fname2>"
    gui = TapeDiffGUI(BasicRenderer(), sys.argv[1], sys.argv[2])
    gui.launch()

"""
GUI for browsing tapes.
"""

import datetime
import logging
import os

import gradio as gr
import uvicorn
import yaml
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from tapeagents.core import Tape, TapeMetadata
from tapeagents.io import load_tapes, stream_yaml_tapes
from tapeagents.observe import retrieve_tape_llm_calls
from tapeagents.renderers.basic import BasicRenderer
from tapeagents.steps import Annotation

logger = logging.getLogger(__name__)


class TapeBrowser:
    """
    Base class for tape browser GUI.
    Displays the list of tapes from the files in the folder, allows to select the file with a bunch of tapes
    and then select tape and view it. Allows to navigate between parent and children tapes by links.
    """

    def __init__(
        self,
        tape_cls: type[Tape],
        tapes_folder: str,
        renderer: BasicRenderer,
        file_extension: str = ".yaml",
    ):
        self.tape_cls = tape_cls
        self.file_extension: str = file_extension
        self.tapes_folder: str = tapes_folder
        self.annotation_file: str = os.path.join(tapes_folder, "annotations.yaml")
        self.renderer: BasicRenderer = renderer
        self.files: list[str] = []
        self.tape_index: dict[str, tuple[int, int]] = {}
        self.tape_children: dict[str, list[str]] = {}
        self.request: gr.Request | None = None
        self.selected_tape: int = 0
        self.tapes: list[Tape] = []
        self.llm_calls: dict = {}

    def load_tapes(self, fname: str) -> list[Tape]:
        fpath = os.path.join(self.tapes_folder, fname)
        try:
            tapes = load_tapes(self.tape_cls, fpath)
            logger.info(f"{len(tapes)} tapes loaded from {fname}")
        except Exception:
            logger.error(f"Could not load tapes from {fpath}")
            raise
        return tapes

    def load_llm_calls(self):
        logger.info("Loading LLM calls")
        self.llm_calls = retrieve_tape_llm_calls(self.tapes)
        logger.info(f"Loaded {len(self.llm_calls)} LLM calls")

    def get_steps(self, tape: Tape) -> list:
        return tape.steps

    def get_context(self, tape: Tape) -> list:
        return getattr(tape.context, "steps", [])

    def get_tape_files(self) -> list[str]:
        files = sorted([f for f in os.listdir(self.tapes_folder) if f.endswith(self.file_extension)])
        assert files, f"No files found in {self.tapes_folder}"
        logger.info(f"{len(files)} files found in {self.tapes_folder}")
        indexed = 0
        nonempty_files = []
        for i, file in enumerate(files):
            if file in self.files:
                continue  # already indexed
            tapes = self.load_tapes(file)
            if not len(tapes):
                logger.warning(f"File {file} does not contain any known tapes, skip")
                continue
            for j, tape in enumerate(tapes):
                tape_id = tape.metadata.id
                parent_id = tape.metadata.parent_id
                if tape_id:
                    if tape_id in self.tape_index and self.tape_index[tape_id] != (i, j):
                        raise ValueError(
                            f"Duplicate tape id {tape_id}. Both in {self.tape_index[tape_id]} and {(i, j)}"
                        )
                    indexed += 1
                    self.tape_index[tape_id] = (i, j)
                    if parent_id:
                        if parent_id not in self.tape_children:
                            self.tape_children[parent_id] = []
                        self.tape_children[parent_id].append(tape_id)
            nonempty_files.append(file)
        logger.info(f"Indexed {indexed} new tapes, index size: {len(self.tape_index)}")
        logger.info(f"{len(self.tape_children)} tapes with children found")
        return nonempty_files

    def get_tape_label(self, tape: Tape) -> str:
        label = ""
        parent_tape_id = tape.metadata.parent_id
        tape_id = tape.metadata.id
        if tape_id:
            label += f"<br><b>ID: {tape_id}</b>"
        if parent_tape_id:
            if parent_tape_id in self.tape_index:
                logger.info(f"Tape index value {self.tape_index[parent_tape_id]}")
                fid, tid = self.tape_index[parent_tape_id]
                logger.info(f"Parent tape {parent_tape_id} found in {self.files[fid]}")
                tape_name = f"{self.files[fid]}/{tid}"
                label += f'<br>Parent Tape: <a href="?tape_id={parent_tape_id}">{tape_name}</a>'
            else:
                label += f"<br>Parent tape: {parent_tape_id}"
        if tape_id and tape_id in self.tape_children:
            children = []
            for cid in self.tape_children[tape_id]:
                tape_name = f"{self.files[self.tape_index[cid][0]]}/{self.tape_index[cid][1]}"
                children.append(f'<div style="margin-left:2em;"><a href="?tape_id={cid}">{tape_name}</a></div>')
            label += f"<br>Children tapes:<br>{''.join(children)}"
        label += f"<br>Length: {len(tape)} steps"
        if tape.metadata:
            m = {k: v for k, v in tape.metadata.model_dump().items() if k not in ["id", "parent_id"]}
            label += f'<h3>Metadata</h3><div style="white-space: pre-wrap;">{yaml.dump(m, allow_unicode=True)}</div>'
        return label

    def get_exp_label(self, filename: str, tapes: list[Tape]) -> str:
        tapelengths = [len(tape) for tape in tapes]
        tapelen = sum(tapelengths) / len(tapelengths)
        return f"<h3>{len(self.tape_index)} indexed tapes in {len(self.files)} files<br>{len(tapes)} tapes in the current file<br>Avg. tape length: {tapelen:.1f} steps</h3>"

    def get_tape_name(self, i: int, tape: Tape) -> str:
        return f"Tape {i}"

    def update_view(self, selected_file: str):
        logger.info(f"Loading tapes from {selected_file}")
        self.tapes = self.load_tapes(selected_file)
        self.load_llm_calls()
        file_label = self.get_exp_label(selected_file, self.tapes)
        tape_names = [(self.get_tape_name(i, tape), i) for i, tape in enumerate(self.tapes)]
        logger.info(f"Selected file: {selected_file}, selected tape: {self.selected_tape}")
        files = gr.Dropdown(self.files, label="Experiment", value=selected_file)  # type: ignore
        tape_names = gr.Dropdown(tape_names, label="Tape", value=self.selected_tape)  # type: ignore
        tape_html, label = self.update_tape_view(self.selected_tape)
        return files, tape_names, file_label, tape_html, label

    def update_tape_view(self, tape_id: int) -> tuple[str, str]:
        logger.info(f"Loading tape {tape_id}")
        if tape_id >= len(self.tapes):
            logger.error(f"Tape {tape_id} not found in the index")
            return f"<h1>Failed to load tape {tape_id}</h1>", ""
        tape = self.tapes[tape_id]
        label = self.get_tape_label(tape)
        html = f"{self.renderer.style}{self.renderer.render_tape(tape, self.llm_calls)}"
        return html, label

    def reload_tapes(self, selected_file: str):
        logger.info(f"Reloading tapes from {selected_file}")
        return self.update_view(selected_file)

    def switch_file(self, selected_file: str):
        logger.info(f"Switching to file {selected_file}")
        self.selected_tape = 0
        return self.update_view(selected_file)

    def update_view_from_request(self, request: gr.Request):
        self.request = request
        tape_id = self.request.query_params.get("tape_id")
        self.files = self.get_tape_files()
        selected_file = self.files[0]
        if tape_id and tape_id in self.tape_index:
            i, j = self.tape_index[tape_id]
            selected_file = self.files[i]
            self.selected_tape = j
            logger.info(f"Selected tape {selected_file}/{j} from query params")
        return self.update_view(selected_file)

    def save_annotation(self, step: int, annotation: str, tape_id: int):
        tape = self.tapes[tape_id]
        time_now = datetime.datetime.now().isoformat()
        annotator_tape = Tape[Tape, Annotation](
            metadata=TapeMetadata(other={"made_at": time_now}),
            context=tape,
            steps=[Annotation(step=step, text=annotation)],
        )
        with stream_yaml_tapes(self.annotation_file, "a") as saver:
            saver.save(annotator_tape)
        gr.Info(f'Saved an annotation to "{self.annotation_file}')

    def create_blocks(self):
        with gr.Blocks(analytics_enabled=False, title="TapeAgents Browser") as blocks:
            with gr.Row():
                with gr.Column(scale=1):
                    file_selector = gr.Dropdown([], label="File")
                    reload_button = gr.Button("Reload Tapes")
                    file_label = gr.HTML("")
                with gr.Column(scale=4):
                    tape_view = gr.HTML("")
                with gr.Column(scale=1):
                    tape_selector = gr.Dropdown([], label="Tape")
                    tape_label = gr.HTML("")
                    gr.HTML("<h2> Annotate tape </h2>")
                    step_selector = gr.Number(0, label="Step")
                    annotation_text = gr.Textbox("", label="Annotation Text", lines=5)
                    annotate_button = gr.Button("Annotate")
            reload_button.click(
                fn=self.reload_tapes,
                inputs=[file_selector],
                outputs=[file_selector, tape_selector, file_label, tape_view, tape_label],
            )
            tape_selector.input(fn=self.update_tape_view, inputs=tape_selector, outputs=[tape_view, tape_label])
            file_selector.input(
                fn=self.switch_file,
                inputs=file_selector,
                outputs=[file_selector, tape_selector, file_label, tape_view, tape_label],
            )
            annotate_button.click(
                fn=self.save_annotation, inputs=[step_selector, annotation_text, tape_selector], outputs=[]
            )
            blocks.load(
                self.update_view_from_request,
                None,
                outputs=[file_selector, tape_selector, file_label, tape_view, tape_label],
            )
        return blocks

    def launch(self, server_name: str = "0.0.0.0", port=7860, debug: bool = False, static_dir: str = ""):
        self.blocks = self.create_blocks()

        if static_dir:
            logger.info(f"Starting FastAPI server with static dir {static_dir}")
            # mount Gradio app to FastAPI app
            app = FastAPI()
            app.mount("/static", StaticFiles(directory=static_dir), name="static")
            app = gr.mount_gradio_app(app, self.blocks, path="/")
            uvicorn.run(app, host=server_name, port=port)
        else:
            self.blocks.launch(server_name=server_name, server_port=port, debug=debug)

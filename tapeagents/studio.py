import logging
from typing import Callable, Generator, Mapping

import gradio as gr
import uvicorn
import yaml
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from tapeagents.agent import Agent
from tapeagents.core import Tape
from tapeagents.environment import Environment
from tapeagents.observe import (
    get_latest_tape_id,
    observe_tape,
    retrieve_tape,
    retrieve_tape_llm_calls,
)
from tapeagents.orchestrator import MainLoopEvent, main_loop
from tapeagents.renderers import render_agent_tree
from tapeagents.renderers.basic import BasicRenderer

logger = logging.getLogger(__name__)


class Studio:
    def __init__(
        self,
        agent: Agent,
        tape: Tape,
        renderers: BasicRenderer | Mapping[str, BasicRenderer],
        environment: Environment | None = None,
        transforms: Mapping[str, Callable[[Tape], Tape]] | None = None,
    ):
        self.renderers = (
            {renderers.__class__.__name__: renderers} if isinstance(renderers, BasicRenderer) else renderers
        )
        self.environment = environment
        self.transforms = transforms or {}
        self.original_agent = agent
        self.original_tape = tape

        with gr.Blocks(title="TapeAgent Studio") as blocks:
            tape_state = gr.State(tape.model_dump())
            agent_state = gr.State(agent.model_dump())
            with gr.Row():
                with gr.Column(scale=1):
                    tape_data = gr.Textbox(
                        "", max_lines=15, label="Raw Tape content", info="Press Enter to rerender the tape"
                    )
                    pop = gr.Number(1, label="Pop N last steps", info="Press Enter to proceed")
                    keep = gr.Number(0, label="Keep N first steps", info="Press Enter to proceed")
                    load = gr.Textbox(max_lines=1, label="Load tape by id", info="Press Enter to load tape")
                    choices = list(self.renderers.keys())
                    renderer_choice = gr.Dropdown(choices=choices, value=choices[0], label="Choose the tape renderer")  # type: ignore
                    if transforms:
                        transform_choice = gr.Dropdown(list(transforms.keys()), label="Run a transform")
                with gr.Column(scale=3):
                    tape_render = gr.HTML("")
                with gr.Column(scale=1):
                    gr.TextArea(render_agent_tree(agent), label="Agents Hierarchy", max_lines=20)
                    agent_config = gr.Textbox(
                        "",
                        max_lines=15,
                        label="Agents configuration",
                        info="Press Enter to update the agent",
                    )
                    run_agent = gr.Button("Run Agent")
                    if environment:
                        run_enviroment = gr.Button("Run Environment")
                        run_loop = gr.Button("Run Loop")
                    else:
                        run_enviroment = None
                        run_loop = None

            render_tape = (self.render_tape, [renderer_choice, tape_state], [tape_data, tape_render])

            blocks.load(*render_tape).then(self.render_agent, [agent_state], [agent_config])

            # Tape controls
            tape_data.submit(
                lambda data: tape.model_validate(yaml.safe_load(data)).with_new_id(),
                [tape_data],
                [tape_state],
            ).then(*render_tape).then(lambda tape_dict: observe_tape(self.validate_tape(tape_dict)), [tape_state], [])
            pop.submit(self.pop_n_steps, [tape_state, pop], [tape_state]).then(*render_tape)
            keep.submit(self.keep_n_steps, [tape_state, keep], [tape_state]).then(*render_tape)
            load.submit(self.load_tape, [tape_state, load], [tape_state]).then(*render_tape)
            renderer_choice.change(*render_tape)
            if transforms:
                transform_choice.change(
                    lambda transform, tape: self.transforms[transform](tape),
                    [transform_choice, tape_state],
                    [tape_state],
                ).then(*render_tape)

            # Agent controls
            agent_config.submit(self.update_agent, [agent_config], [agent_state]).then(lambda: gr.Info("Agent updated"))
            run_agent.click(
                self.run_agent, [renderer_choice, agent_state, tape_state], [tape_state, tape_data, tape_render]
            )
            if environment:
                assert run_enviroment and run_loop
                run_enviroment.click(self.run_environment, [tape_state], [tape_state]).then(*render_tape)
                run_loop.click(
                    self.run_main_loop, [renderer_choice, agent_state, tape_state], [tape_state, tape_data, tape_render]
                )

        self.blocks = blocks

    def validate_tape(self, tape_dict: dict) -> Tape:
        return self.original_tape.model_validate(tape_dict)

    def pop_n_steps(self, tape_dict: dict, n: int) -> Tape:
        tape = self.validate_tape(tape_dict)
        if n > len(tape):
            raise gr.Error(f"Cannot pop {n} steps from tape with {len(tape)} steps")
        return tape[:-n]

    def keep_n_steps(self, tape_dict: dict, n: int) -> Tape:
        tape = self.validate_tape(tape_dict)
        if n > len(tape):
            raise gr.Error(f"Cannot keep {n} steps from tape with {len(tape)} steps")
        return tape[:n]

    def transform_tape(self, transform: str, tape_dict: dict) -> Tape:
        tape = self.validate_tape(tape_dict)
        result = self.transforms[transform](tape)
        observe_tape(result)
        return result

    def load_tape(self, tape_dict: dict, tape_id: str) -> Tape:
        tape = self.validate_tape(tape_dict)
        if not tape_id:
            tape_id = get_latest_tape_id()
        result = retrieve_tape(type(tape), tape_id)
        if not result:
            raise gr.Error(f"No tape found with id {tape_id}")
        return result

    def render_tape(self, renderer_name: str, tape_dict: dict) -> tuple[str, str]:
        tape = self.validate_tape(tape_dict)
        renderer = self.renderers[renderer_name]
        llm_calls = retrieve_tape_llm_calls(tape)
        return (
            yaml.safe_dump(tape.model_dump(), sort_keys=False),
            renderer.style + renderer.render_tape(tape, llm_calls),
        )

    def validate_agent(self, agent_dict: dict) -> Agent:
        return self.original_agent.update(agent_dict)

    def render_agent(self, agent_dict: dict) -> str:
        agent = self.validate_agent(agent_dict)
        try:
            agent_config = yaml.dump(agent.model_dump(), sort_keys=False)
        except Exception as e:
            logger.exception(f"Failed to get agent configuration: {e}")
            agent_config = "Failed to get agent configuration"
        return agent_config

    def update_agent(self, config: str) -> dict:
        return self.original_agent.update(yaml.safe_load(config)).model_dump()

    def run_agent(
        self, renderer_name: str, agent_dict: dict, tape_dict: dict
    ) -> Generator[tuple[Tape, str, str], None, None]:
        agent = self.validate_agent(agent_dict)
        start_tape = self.validate_tape(tape_dict)
        for event in agent.run(start_tape):
            if tape := event.partial_tape or event.final_tape:
                observe_tape(tape)
                yield (tape,) + self.render_tape(renderer_name, tape)

    def run_environment(self, tape: Tape) -> Tape:
        assert self.environment
        start_tape = self.validate_tape(tape)
        return self.environment.react(start_tape)

    def run_main_loop(
        self, renderer_name: str, agent_dict: dict, tape_dict: dict
    ) -> Generator[tuple[Tape, str, str], None, None]:
        assert self.environment
        agent = self.validate_agent(agent_dict)
        start_tape = self.validate_tape(tape_dict)
        last_tape = start_tape
        for event in main_loop(agent, start_tape, self.environment):
            if ae := event.agent_event:
                if ae.step:
                    logger.info(f"added step {type(ae.step).__name__}")
                if tape := ae.partial_tape or ae.final_tape:
                    observe_tape(tape)
                    last_tape = tape
                    yield (tape,) + self.render_tape(renderer_name, tape)
            elif event.observation:
                last_tape = last_tape.append(event.observation)
                yield (last_tape,) + self.render_tape(renderer_name, last_tape)
            elif isinstance(event, MainLoopEvent):
                pass
            else:
                raise ValueError("Unexpected event", event)

    def launch(
        self, server_name: str = "0.0.0.0", port=7860, debug: bool = False, static_dir: str = "", *args, **kwargs
    ):
        if static_dir:
            logger.info(f"Starting FastAPI server with static dir {static_dir}")
            # mount Gradio app to FastAPI app
            app = FastAPI()
            app.mount("/static", StaticFiles(directory=static_dir), name="static")
            app = gr.mount_gradio_app(app, self.blocks, path="/")
            uvicorn.run(app, host=server_name, port=port)
        else:
            self.blocks.launch(server_name=server_name, debug=debug, *args, **kwargs)

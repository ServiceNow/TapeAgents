import logging
from collections import defaultdict
from typing import Generator

import gradio as gr

from tapeagents.agent import Agent, Annotator, ObservationMaker
from tapeagents.core import Episode, Tape
from tapeagents.dialog_tape import AssistantStep, UserStep
from tapeagents.environment import Environment, ExternalObservationNeeded
from tapeagents.orchestrator import main_loop
from tapeagents.renderers.basic import BasicRenderer

logger = logging.getLogger(__name__)


class Demo:
    def __init__(
        self,
        agent: Agent,
        start_tape: Tape,
        environment: Environment,
        renderer: BasicRenderer,
        annotator: Annotator | None = None,
        user_models: dict[str, ObservationMaker] | None = None,
        default_input: str = "",
    ):
        self.agent = agent
        self.environment = environment
        self.renderer = renderer
        self.annotator = annotator
        self.user_models = user_models

        with gr.Blocks() as blocks:
            tape = gr.State(start_tape)
            user_tapes = gr.State({})
            annotator_tapes = gr.State({})
            with gr.Column():
                chat = gr.HTML(self.render_episode(start_tape, {}, {}))
                with gr.Row():
                    input = gr.Textbox(label="Type your next message here", value=default_input)
                    if user_models:
                        with gr.Column():
                            um_names: list = list(user_models.keys())
                            user_model_list = gr.Radio(
                                choices=um_names,
                                value=um_names[0],
                                label="Choose user model",
                            )
                            generate_user_message = gr.Button("Generate user message")

            input.submit(
                self.add_user_step,
                inputs=[input, tape],
                outputs=[tape],
            ).then(self.render_episode, inputs=[tape, annotator_tapes, user_tapes], outputs=[chat]).then(
                self.run_main_loop, inputs=[tape, annotator_tapes, user_tapes], outputs=[tape, annotator_tapes, chat]
            )

            if user_models:
                generate_user_message.click(
                    self.generate_user_step,
                    inputs=[user_model_list, tape, user_tapes],
                    outputs=[tape, user_tapes],
                ).then(self.render_episode, inputs=[tape, annotator_tapes, user_tapes], outputs=[chat]).then(
                    self.run_main_loop,
                    inputs=[tape, annotator_tapes, user_tapes],
                    outputs=[tape, annotator_tapes, chat],
                )
        self.blocks = blocks

    def launch(self, *args, **kwargs):
        self.blocks.launch(*args, **kwargs)

    def render_episode(self, tape: Tape, annotator_tapes: dict[int, list[Tape]], user_tapes: dict[int, Tape]) -> str:
        css = self.renderer.style
        html = self.renderer.render_episode(
            Episode(tape=tape, annotator_tapes=annotator_tapes, obs_making_tapes=user_tapes)
        )
        return css + html

    def add_user_step(self, user_input: str, tape: Tape) -> Tape:
        return tape.append(UserStep(content=user_input))

    def generate_user_step(
        self, user_model_name: str, tape: Tape, user_tapes: dict[int, Tape]
    ) -> tuple[Tape, dict[int, Tape]]:
        assert self.user_models
        user_model = self.user_models[user_model_name]
        for event in user_model.run(user_model.make_own_tape(tape)):
            if own_tape := event.final_tape:
                new_tape = user_model.add_observation(tape, own_tape)
                user_tapes[len(tape)] = own_tape
                return new_tape, user_tapes
        raise ValueError()

    def run_main_loop(
        self, start_tape: Tape, annotator_tapes: dict[int, list[Tape]], user_tapes: dict[int, Tape]
    ) -> Generator[tuple[Tape, dict, str], None, None]:
        annotator_tapes = defaultdict(list, annotator_tapes)
        try:
            for event in main_loop(self.agent, start_tape, self.environment):
                if ae := event.agent_event:
                    if ae.partial_tape:
                        logger.info(f"added step {type(ae.step).__name__}")
                        tape = ae.partial_tape
                    elif ae.final_tape:
                        tape = ae.final_tape
                        logger.info("received final tape")
                    else:
                        # TODO: handle partial steps
                        continue
                    # TODO: configure at which step the annotator is called
                    yield tape, annotator_tapes, self.render_episode(tape, annotator_tapes, user_tapes)
                    if isinstance(ae.step, AssistantStep) and self.annotator:
                        annotator_tape = self.annotator.annotate(tape)
                        annotator_tapes[len(tape) - 1].append(annotator_tape)
                        logger.info(f"made an annotation for step {len(tape) - 1}")
                        yield tape, annotator_tapes, self.render_episode(tape, annotator_tapes, user_tapes)
                if obs := event.observation:
                    tape = tape.append(obs)
                yield tape, annotator_tapes, self.render_episode(tape, annotator_tapes, user_tapes)
        except ExternalObservationNeeded as e:
            assert isinstance(e.action, AssistantStep)
            logger.info("main loop finished, waiting for the next user message")

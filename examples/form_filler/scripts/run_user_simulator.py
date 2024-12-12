import json
import logging
import traceback
from pathlib import Path

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
from tqdm import tqdm

from tapeagents.io import load_tapes, stream_yaml_tapes
from tapeagents.parallel_processing import lazy_thread_pool_processor

from ..tape import FormFillerAgentMetadata, FormFillerTape, FormFillerUserMetadata
from ..user_simulator_agent import (
    SampleUserInstructionThought,
    UserSimulatorAgent,
    UserSimulatorError,
    UserSimulatorTape,
)

logger = logging.getLogger(__name__)


def run_user_simulator_agent(
    form_filler_tape: FormFillerTape, user_simulator_agent: UserSimulatorAgent
) -> tuple[None | Exception, FormFillerTape, None | UserSimulatorTape]:
    # assert this tape is a user tape or user_simulator_agent tape
    assert form_filler_tape.context is not None, "Tape context is None"
    assert isinstance(
        form_filler_tape.metadata, (FormFillerAgentMetadata)
    ), "Tape metadata is not of type FormFillerUserMetadata or FormFillerAgentMetadata"

    user_simulator_agent_tape = None
    try:
        # Convert formfiller tape to UserSimulatorAgent tape
        user_simulator_agent_tape = user_simulator_agent.make_own_tape(form_filler_tape)

        user_simulator_agent_tape = user_simulator_agent.run(user_simulator_agent_tape).get_final_tape()

        # Reconvert to formfiller tape
        continued_tape = user_simulator_agent.add_observation(form_filler_tape, user_simulator_agent_tape)
        # fix the metadata to be the correct type
        continued_tape.metadata = FormFillerUserMetadata(
            **continued_tape.metadata.model_dump(),
            last_agent_tape_id=form_filler_tape.metadata.id,
        )
        # recover the primary and secondary user behaviors used
        continued_tape.metadata.user_behavior = user_simulator_agent.behavior_alias
        if isinstance(sample_step := user_simulator_agent_tape.steps[-2], SampleUserInstructionThought):
            continued_tape.metadata.user_secondary_behavior = sample_step.instruction_alias

    except Exception as e:
        # Return input formfiller tape
        if user_simulator_agent_tape is not None:  # if make_own_tape didn't fail
            user_simulator_agent_tape.steps.append(UserSimulatorError(error=str(e) + "\n" + traceback.format_exc()))

        return e, form_filler_tape, user_simulator_agent_tape
    return None, continued_tape, user_simulator_agent_tape


@hydra.main(config_path="../conf", config_name="run_user_simulator", version_base="1.2")
def main(cfg: DictConfig):
    output_path = Path(cfg.output_path)  # must be a directory
    output_path.mkdir(exist_ok=True, parents=True)

    logger.info(f"Hydra runtime dir: {hydra.core.hydra_config.HydraConfig.get().runtime.output_dir}")
    logger.info(f"Output path: {output_path}")

    input_dialogues_path = cfg.input_dialogues_path

    tapes: list[FormFillerTape] = load_tapes(FormFillerTape, input_dialogues_path)
    user_simulator_agent = instantiate(cfg.user_simulator_agent)

    valid_tapes: list[FormFillerTape] = []
    for tape in tapes:
        if user_simulator_agent.can_continue(tape):
            valid_tapes.append(tape)
        else:
            logger.debug(f"Skipping tape {tape.metadata.id} as it is not continuable")
    logger.info(f"Continuable tapes: {len(valid_tapes)}/{len(tapes)} total tapes")

    predicted_tapes: list[FormFillerTape] = []
    failed_tapes = []
    user_simulator_tapes = []
    for i, (exception, form_filler_tape, user_simulator_tape) in enumerate(
        lazy_thread_pool_processor(
            stream=tqdm(
                valid_tapes,
                total=min(len(valid_tapes), cfg.max_continuable_tapes)
                if cfg.max_continuable_tapes >= 0
                else len(valid_tapes),
                desc="Running user simulator agent",
            ),
            worker_func=lambda tape: run_user_simulator_agent(tape, user_simulator_agent),
            n_workers=cfg.n_workers,
        )
    ):
        if cfg.max_continuable_tapes >= 0 and i >= cfg.max_continuable_tapes:
            logger.warning(f"Reached max_continuable_tapes limit of {cfg.max_continuable_tapes}")
            break

        user_simulator_tapes.append(user_simulator_tape)

        if exception is not None:
            logger.error(f"Tape {form_filler_tape.metadata.id} failed with error: {exception}")
            logger.exception(exception, exc_info=exception)
            failed_tapes.append(form_filler_tape)

        else:
            logger.debug(
                f"Successfully continued tape {form_filler_tape.metadata.id} with {form_filler_tape.metadata.n_added_steps} steps"
            )
            predicted_tapes.append(form_filler_tape)

    counters = {
        "total_input_tapes": len(tapes),
        "total_continuable_tapes": len(valid_tapes),
        "total_user_simulator_tapes": len(user_simulator_tapes),
        "total_predicted_tapes": len(predicted_tapes),
        "total_failed_tapes": len(failed_tapes),
    }
    logger.info(json.dumps(counters, indent=2))
    with open(output_path / "counters.json", "w") as f:
        json.dump(counters, f, indent=2)

    with stream_yaml_tapes(output_path / "user_simulator_tapes.yaml") as saver:
        for tape in tqdm(user_simulator_tapes, desc="Saving user simulator tapes"):
            saver.save(tape)

    with stream_yaml_tapes(output_path / "user_predicted_tapes.yaml") as saver:
        for tape in tqdm(predicted_tapes, desc="Saving predicted tapes"):
            saver.save(tape)

    if failed_tapes:
        with stream_yaml_tapes(output_path / "user_failed_tapes.yaml") as saver:
            for tape in tqdm(failed_tapes, desc="Saving failed tapes"):
                saver.save(tape)


if __name__ == "__main__":
    main()

import json
import logging
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Generator

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
from tqdm import tqdm

from tapeagents.agent import Agent
from tapeagents.core import Error, TapeMetadata
from tapeagents.io import load_tapes, stream_yaml_tapes
from tapeagents.orchestrator import main_loop
from tapeagents.parallel_processing import lazy_thread_pool_processor

from ..environment import FormFillerEnvironment
from ..error import UnknownError
from ..student import StudentAgent
from ..tape import (
    FormFillerAgentMetadata,
    FormFillerContext,
    FormFillerTape,
    FormFillerUserMetadata,
)

logger = logging.getLogger(__name__)


def run_formfiller_agent(
    tape: FormFillerTape, agent: Agent[FormFillerTape]
) -> tuple[FormFillerTape, FormFillerTape | Exception]:
    """
    Run the agent on the given tape
    Returns the input tape and (the predicted tape or an exception)
    """
    try:
        # assert this tape is a user tape
        assert isinstance(
            tape.metadata, FormFillerUserMetadata
        ), f"Expected FormFillerUserMetadata as input tape, got {type(tape.metadata)}. Tape: {tape.metadata.id}"
        # create environment
        assert isinstance(
            tape.context, FormFillerContext
        ), f"Expected FormFillerContext, got {type(tape.context)}. Tape: {tape.metadata.id}"
        env = FormFillerEnvironment.from_spec(tape.context.env_spec)
        # run main_loop
        predicted_tape = main_loop(agent, tape, env).get_final_tape()
        # update metadata accordingly
        assert isinstance(predicted_tape.metadata, TapeMetadata)
        predicted_tape.metadata = FormFillerAgentMetadata(
            **predicted_tape.metadata.model_dump(),
            last_user_tape_id=tape.metadata.id,
        )
        # update context
        assert isinstance(predicted_tape.context, FormFillerContext)
        predicted_tape.context.date = datetime.now().strftime("%Y-%m-%d")
        return tape, predicted_tape
    except Exception as e:
        return tape, e


@hydra.main(version_base=None, config_path="../conf", config_name="run_formfiller_agent")
def main(cfg: DictConfig):
    agent = instantiate(cfg.agent)

    agent_type = isinstance(agent, StudentAgent) and "student" or "teacher"
    logger.info(f"Agent type: {agent_type}")

    user_dialogues_path = cfg.user_dialogues_path
    tapes = load_tapes(FormFillerTape, user_dialogues_path)

    def tape_gen() -> Generator[FormFillerTape, None, None]:
        for tape in tqdm(tapes):
            yield tape  # type: ignore

    predicted_tapes: list[FormFillerTape] = []
    failure_tapes: list[FormFillerTape] = []
    for result in lazy_thread_pool_processor(
        stream=tape_gen(),
        worker_func=lambda tape: run_formfiller_agent(tape, agent),
        n_workers=cfg.n_workers,
    ):
        if isinstance(result, Exception):
            logger.exception(f"Error while running lazy_thread_pool_processor: {result}", exc_info=result)
            continue
        else:
            input_tape, result = result
            if isinstance(result, Exception):
                logger.exception(f"Error while running run_formfiller_agent: {result}", exc_info=result)
                predicted_tape = input_tape.model_copy(
                    update=dict(steps=input_tape.steps + [UnknownError(message=str(result))])
                )
            else:
                predicted_tape = result

        if predicted_tape.last_action is None or any(isinstance(step, Error) for step in predicted_tape.steps):
            failure_tapes.append(predicted_tape)
        else:
            predicted_tapes.append(predicted_tape)

    output_path = Path(cfg.output_path)  # must be a directory
    output_path.mkdir(exist_ok=True, parents=True)

    # save predicted tapes
    stats = Counter()
    with stream_yaml_tapes(output_path / f"{agent_type}_predicted_tapes.yaml") as saver:
        for tape in tqdm(predicted_tapes, desc="Saving predicted tapes"):
            stats[tape.last_action.kind] += 1
            saver.save(tape)
    print(stats)

    # save failure tapes
    error_stats = Counter()
    if failure_tapes:
        with stream_yaml_tapes(output_path / f"{agent_type}_failure_tapes.yaml") as saver:
            for tape in tqdm(failure_tapes, desc="Saving failure tapes"):
                error_stats[tape.last_action.kind] += 1
                saver.save(tape)
        print(error_stats)

    # save stats: counters of last step kinds
    with open(output_path / f"{agent_type}_stats.json", "w") as f:
        json.dump(
            {
                "predicted_tapes_stats": stats,
                "failure_tapes_stats": error_stats,
            },
            f,
            indent=2,
        )


if __name__ == "__main__":
    main()

import json
import logging
import math
import random
import shutil
import traceback
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Generator, Literal

import hydra
from hydra.core.global_hydra import GlobalHydra
from hydra.utils import instantiate
from omegaconf import DictConfig
from pydantic import BaseModel
from tqdm import tqdm

from tapeagents.agent import Agent
from tapeagents.core import Error
from tapeagents.dialog_tape import AssistantStep
from tapeagents.io import load_tapes, stream_yaml_tapes
from tapeagents.parallel_processing import lazy_thread_pool_processor

from ..error import UnknownError
from ..tape import FormFillerAgentMetadata, FormFillerContext, FormFillerTape
from ..user_simulator_agent import UserSimulatorAgent, UserSimulatorTape
from .run_formfiller_agent import run_formfiller_agent
from .run_user_simulator import run_user_simulator_agent

logger = logging.getLogger(__name__)


def export_merged_dialogues(output_path: Path, num_layer):
    output_path = Path(output_path)
    logger.info(f"Exporting merged layers from {output_path}")
    all_user_tapes: list[UserSimulatorTape] = []
    all_formfiller_user_tapes: list[FormFillerTape] = []
    all_formfiller_agent_successes: list[FormFillerTape] = []
    all_formfiller_agent_failures: list[FormFillerTape] = []
    for i in range(num_layer):
        logger.info(f"Processing layer {i}")

        if i % 2 == 0:  # user layer
            all_formfiller_user_tapes.extend(
                load_tapes(FormFillerTape, output_path / f"layer_{i}" / "data.yaml")  # type: ignore
            )

            all_user_tapes.extend(
                load_tapes(UserSimulatorTape, output_path / f"layer_{i}" / "user_simulator_tapes.yaml")  # type: ignore
            )
        else:  # agent layer
            all_formfiller_agent_successes.extend(
                load_tapes(FormFillerTape, output_path / f"layer_{i}" / "data.yaml")  # type: ignore
            )

            failures_path = output_path / f"layer_{i}" / "failures.yaml"
            if failures_path.exists():
                all_formfiller_agent_failures.extend(
                    load_tapes(FormFillerTape, failures_path)  # type: ignore
                )

    stats = {
        "formfiller_agent_tapes.yaml": Counter(),
        "formfiller_user_tapes.yaml": Counter(),
        "user_simulator_tapes.yaml": Counter(),
    }
    logger.info(
        f'Saving {len(all_formfiller_agent_successes)} successful agent forks to {output_path / "formfiller_agent_tapes.yaml"}'
    )
    with stream_yaml_tapes(output_path / "formfiller_agent_tapes.yaml") as saver:
        for tape in all_formfiller_agent_successes:
            stats["formfiller_agent_tapes.yaml"][tape.last_action.kind] += 1
            saver.save(tape)

    if all_formfiller_agent_failures:
        stats["formfiller_agent_tape_failures.yaml"] = Counter()
        logger.info(
            f'Saving {len(all_formfiller_agent_failures)} failed agent forks to {output_path / "formfiller_agent_tape_failures.yaml"}'
        )
        with stream_yaml_tapes(output_path / "formfiller_agent_tape_failures.yaml") as saver:
            for tape in all_formfiller_agent_failures:
                stats["formfiller_agent_tape_failures.yaml"][tape.last_action.kind] += 1
                saver.save(tape)

    logger.info(f'Saving {len(all_formfiller_user_tapes)} user forks to {output_path / "formfiller_user_tapes.yaml"}')
    with stream_yaml_tapes(output_path / "formfiller_user_tapes.yaml") as saver:
        for tape in all_formfiller_user_tapes:
            stats["formfiller_user_tapes.yaml"][tape.steps[-1].kind] += 1
            saver.save(tape)

    logger.info(f'Saving {len(all_user_tapes)} user simulator tapes to {output_path / "user_simulator_tapes.yaml"}')
    with stream_yaml_tapes(output_path / "user_simulator_tapes.yaml") as saver:
        for tape in all_user_tapes:
            stats["user_simulator_tapes.yaml"][tape.steps[-1].kind] += 1
            saver.save(tape)

    logger.info(f'Saving stats to {output_path / "stats.json"}')
    with open(output_path / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)


class Layer(BaseModel):
    """
    Layer definitions for user and agent
    """

    who: Literal["user", "agent"]
    count: int
    probs: dict[str, float] = {}


def delete_layer_dialogues(output_path: Path, layer_idx: int):
    # Remove the folder with rmtree
    layer_path = Path(output_path) / f"layer_{layer_idx}"
    if layer_path.exists():
        logger.info(f"-> Deleting layer {layer_idx}")
        shutil.rmtree(layer_path)
        return True
    return False


def resume_last_completed_layer(output_path: Path, force_restart_idx: int = -1) -> tuple[int, list[FormFillerTape]]:
    """
    Return the last completed layer's dialogues and index
    Layers are marked as done by creating a file named 'DONE' in the layer folder.

    If restart_layer_idx >= 0, delete all layers >= force_restart_idx.
    Use restart_layer_idx = -1 to resume from the last finished layer.
    """
    if force_restart_idx >= 0:
        logger.info(f"Deleting all layers >= {force_restart_idx} from {output_path}")
        # Delete all layers
        for layer_idx in range(force_restart_idx, 10000):
            if not delete_layer_dialogues(output_path, layer_idx):
                break

    last_completed_layer_dialogues: list[FormFillerTape] = []
    last_completed_layer_idx = -1
    logger.info(f"Attempting to resume dialogues from {output_path}")
    while True:
        if not (Path(output_path) / f"layer_{last_completed_layer_idx + 1}" / "DONE").exists():
            break
        last_completed_layer_idx += 1
        logger.info(f"-> Layer {last_completed_layer_idx} is done, skipping")
    logger.info(f"-> Last completed layer: {last_completed_layer_idx}")
    if last_completed_layer_idx >= 0:
        # Load dialogues from last completed layer
        last_completed_layer_dialogues = load_tapes(
            FormFillerTape, Path(output_path) / f"layer_{last_completed_layer_idx}" / "data.yaml"
        )
    return last_completed_layer_idx, last_completed_layer_dialogues


def parse_layers(layer_objs: list[dict[str, Any]], global_count: int = -1) -> list[Layer]:
    """
    Parse list of layers from a list of dictionaries.
    For user layers, replace "probs: same" with the probabilities of the previous user layer.
    """
    layers = []
    previous_user_layer = None
    previous_layer = None
    for layer_obj in layer_objs:
        if layer_obj.get("probs") == "same":
            assert layer_obj["who"] == "user", "copy_probs is only supported for user layers"
            assert previous_user_layer is not None, "No previous user layer to copy probabilities from"
            layer_obj["probs"] = previous_user_layer.probs
        if global_count >= 0:
            layer_obj["count"] = global_count
        elif layer_obj.get("count") == "same":
            assert previous_layer is not None, "No previous layer to copy count from"
            layer_obj["count"] = previous_layer.count

        layer = Layer.model_validate(layer_obj)
        layers.append(layer)

        previous_layer = layer
        if layer.who == "user":
            previous_user_layer = layer
    return layers


def make_tape_tree(
    *,
    agent: Agent[FormFillerTape],
    layers: list[Layer],
    user_simulator_agents: dict[str, UserSimulatorAgent],
    env_spec: str,
    previous_layer_idx: int = -1,  # -1 if no previous layer
    previous_layer_dialogues: list[FormFillerTape] = [],  # must be empty at first layer
    num_workers: int = 0,
) -> Generator[tuple[int, list[FormFillerTape], list[FormFillerTape], list[UserSimulatorTape] | None], None, None]:
    if previous_layer_idx == -1:
        assert not previous_layer_dialogues, "previous_layer_dialogues must be empty at the first layer"

        # Generate a single empty dialogue
        previous_layer_dialogues = [
            FormFillerTape(
                metadata=FormFillerAgentMetadata(
                    author="make_tape_tree",
                ),
                context=FormFillerContext(
                    env_spec=env_spec,
                    date=str(datetime.now().date()),
                ),
                steps=[AssistantStep(content="Hi, how can I help you?")],
            )
        ]
    else:
        assert previous_layer_idx >= 0, "previous_layer_idx must be >= 0"

    current_layer_idx = previous_layer_idx + 1

    while current_layer_idx < len(layers):
        assert previous_layer_dialogues, "previous_layer_dialogues must not be empty at previous layers"

        layer = layers[current_layer_idx]
        current_layer_dialogues = []
        current_layer_failures = []

        logger.info(f"*************** Processing Layer {current_layer_idx} ***************")

        if layer.who == "user":
            # (1) get all the tapes that each user behavior can continue
            continuable_tapes_per_behavior = {}
            for behavior in layer.probs:
                user_simulator_agent = user_simulator_agents[behavior]

                # Get valid tapes to continue
                continuable_tapes: list[FormFillerTape] = []
                for tape in previous_layer_dialogues:
                    if user_simulator_agent.can_continue(tape):
                        continuable_tapes.append(tape)
                    else:
                        pass  # logger.debug(f"Skipping tape {tape.metadata.id} as it is not continuable")
                logger.info(
                    f"Continuable tapes for behavior {behavior}: {len(continuable_tapes)}/{len(previous_layer_dialogues)} total tapes"
                )

                random.shuffle(continuable_tapes)  # shuffle in place

                continuable_tapes_per_behavior[behavior] = continuable_tapes
            assert (
                sum(map(len, continuable_tapes_per_behavior.values())) > 0
            ), "No continuable tapes found for any behavior"

            # (2) compute the number of tapes to continue for each behavior based on the config probs
            continuable_counts = {
                behavior: prob for behavior, prob in layer.probs.items() if continuable_tapes_per_behavior[behavior]
            }
            behavior_counts = {
                behavior: math.ceil(layer.count * prob / sum(continuable_counts.values()))
                for behavior, prob in continuable_counts.items()
            }
            logger.info(f"Will continue these behaviors with these counts: {behavior_counts}")
            logger.info(
                f"These behaviors could not be continued: {set(layer.probs.keys()) - set(behavior_counts.keys())}"
            )

            # (3) generate the user simulator tapes
            user_simulator_tapes = []
            for behavior, behavior_count in behavior_counts.items():
                continuable_tapes = continuable_tapes_per_behavior[behavior]

                user_simulator_agent = user_simulator_agents[behavior]

                # Just repeat them to fill up the count
                # TODO: limit the number of UF/AF
                continuable_tapes = continuable_tapes * math.ceil(behavior_count / len(continuable_tapes))
                logger.info(f"Calling run_user_simulator_agent for behavior {behavior} with count {behavior_count}")
                logger.info(f"Continuable tapes after repeating: {len(continuable_tapes)}")

                try:
                    for i, (exception, form_filler_tape, user_simulator_tape) in enumerate(
                        lazy_thread_pool_processor(
                            stream=tqdm(continuable_tapes, desc=f"Making User Layer with behavior {behavior}"),
                            worker_func=lambda tape: run_user_simulator_agent(tape, user_simulator_agent),
                            n_workers=num_workers,
                        )
                    ):
                        if i >= behavior_count:
                            logger.warning(f"Reached limit of behavior_count={behavior_count}")
                            break

                        user_simulator_tapes.append(user_simulator_tape)

                        if exception is not None:
                            logger.error(f"Tape {form_filler_tape.metadata.id} failed with error: {exception}")
                            logger.exception(exception, exc_info=exception)
                            current_layer_failures.append(form_filler_tape)
                        elif any(
                            isinstance(step, Error) for step in form_filler_tape.steps
                        ):  # should not happen since input formfiller_tapes are valid and we only add a UserStep
                            logger.error(f"Tape {form_filler_tape.metadata.id} had LLM parsing failure")
                            current_layer_failures.append(form_filler_tape)
                        else:
                            logger.debug(
                                f"Successfully continued tape {form_filler_tape.metadata.id} with {form_filler_tape.metadata.n_added_steps} steps"
                            )
                            current_layer_dialogues.append(form_filler_tape)

                except Exception:
                    logger.error(f"Skipped behavior {behavior}")
                    traceback.print_exc()

            yield current_layer_idx, current_layer_dialogues, current_layer_failures, user_simulator_tapes

        elif layer.who == "agent":
            for result in lazy_thread_pool_processor(
                stream=tqdm(previous_layer_dialogues, desc="Making Agent Layer"),
                worker_func=lambda tape: run_formfiller_agent(tape, agent),
                n_workers=num_workers,
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

                if any(isinstance(step, Error) for step in predicted_tape.steps):
                    current_layer_failures.append(predicted_tape)
                    logger.error(f"Agent failed to complete tape {predicted_tape.metadata.id}")
                else:
                    current_layer_dialogues.append(predicted_tape)
                    logger.debug(
                        f"Agent successfully completed tape {predicted_tape.metadata.id} with {predicted_tape.metadata.n_added_steps} steps"
                    )

            yield current_layer_idx, current_layer_dialogues, current_layer_failures, None

        else:
            raise ValueError(f"Unexpected layer.who: {layer.who}")

        previous_layer_dialogues = current_layer_dialogues
        current_layer_idx += 1


def main(cfg: DictConfig):
    # Initialize the agent
    agent = instantiate(cfg.agent)

    if agent.llm.parameters["temperature"] < 0.2:
        raise ValueError("Completion engine temperature must be >= 0.2")

    # Parse all layer specifications
    layers = parse_layers(cfg.layers, cfg.global_count)
    num_workers: int = cfg.num_workers

    # Instantiate all user simulator agents
    logger.info("Instantiating all user simulator agents")
    user_simulator_agents: dict[str, UserSimulatorAgent] = {}
    GlobalHydra.instance().clear()
    hydra.initialize(config_path="../conf", version_base="1.2")
    for layer in layers:
        if layer.who == "user":
            for behavior_name in layer.probs:
                if behavior_name in user_simulator_agents:
                    continue
                my_cfg = hydra.compose(
                    overrides=[
                        f"+user_simulator_agent={behavior_name}",
                        f"+llm@user_simulator_agent.llms={cfg.user_simulator_llm}",
                    ],
                )
                user_simulator_agents[behavior_name] = instantiate(my_cfg.user_simulator_agent)
                if user_simulator_agents[behavior_name].llm.parameters["temperature"] < 0.2:  # type: ignore
                    raise ValueError("User completion engine temperature must be >= 0.2")

    logger.info(f"-> {len(user_simulator_agents)} user simulator agents instantiated")
    logger.info(user_simulator_agents.keys())

    # Reload previous dialogues (layers). Will delete layers if force_restart_idx >= 0
    # previous_layer_idx is the last fully processed layer
    previous_layer_idx, previous_layer_dialogues = resume_last_completed_layer(cfg.output_path, cfg.force_restart_idx)

    for current_layer_idx, current_layer_dialogues, current_layer_failures, user_simulator_tapes in make_tape_tree(
        agent=agent,
        layers=layers,
        user_simulator_agents=user_simulator_agents,
        previous_layer_idx=previous_layer_idx,
        previous_layer_dialogues=previous_layer_dialogues,
        num_workers=num_workers,
        env_spec=cfg.preambles_path,
        # Make sure `preambles_path` matches before_summary.env_spec of the previous layers or risk encountering unexpected behaviors
    ):
        # Save tapes
        logger.info(f"Saving dialogues from layer {current_layer_idx} at {cfg.output_path}")
        save_layer_dialogues(
            cfg.output_path, current_layer_idx, current_layer_dialogues, current_layer_failures, user_simulator_tapes
        )

    export_merged_dialogues(cfg.output_path, num_layer=len(layers))


def save_layer_dialogues(
    output_path: Path,
    layer_idx: int,
    dialogues: list[FormFillerTape],
    failures: list[FormFillerTape],
    user_simulator_tapes: list[UserSimulatorTape] | None = None,
):
    layer_path = Path(output_path) / f"layer_{layer_idx}"
    layer_path.mkdir(parents=True, exist_ok=True)

    if dialogues:
        with stream_yaml_tapes(layer_path / "data.yaml") as saver:
            for dialogue in tqdm(dialogues, desc=f"Saving dialogues for layer {layer_idx}"):
                saver.save(dialogue)

    if failures:
        with stream_yaml_tapes(layer_path / "failures.yaml") as saver:
            for dialogue in tqdm(failures, desc=f"Saving failures for layer {layer_idx}"):
                saver.save(dialogue)

    if user_simulator_tapes:
        with stream_yaml_tapes(layer_path / "user_simulator_tapes.yaml") as saver:
            for tape in tqdm(user_simulator_tapes, desc=f"Saving user simulator tapes for layer {layer_idx}"):
                saver.save(tape)

    with open(layer_path / "DONE", "w") as f:
        f.write("")


@hydra.main(version_base="1.2", config_path="../conf", config_name="make_tape_tree")
def main_wrapper(cfg: DictConfig):
    main(cfg.make_tape_tree)


if __name__ == "__main__":
    main_wrapper()

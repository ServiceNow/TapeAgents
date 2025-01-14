"""Batch processing of tapes."""

import logging
import random
import traceback
from functools import partial
from pathlib import Path
from typing import Generator, Generic, Sequence

from pydantic import BaseModel

from tapeagents.agent import Agent, Annotator, ObservationMaker
from tapeagents.config import is_debug_mode
from tapeagents.core import AnnotatorTapeType, ObservationMakerTapeType, Tape, TapeMetadata, TapeType
from tapeagents.environment import Environment
from tapeagents.io import stream_yaml_tapes
from tapeagents.orchestrator import main_loop
from tapeagents.parallel_processing import choose_processor

logger = logging.getLogger(__name__)

_DEFAULT_N_WORKERS = 16


def batch_main_loop(
    agent: Agent[TapeType],
    tapes: list[TapeType],
    environments: Environment | list[Environment],
    n_workers: int = _DEFAULT_N_WORKERS,
    strict: bool = False,
    max_loops: int = -1,
) -> Generator[TapeType, None, None]:
    """Continue tapes in parallel using an agent."""
    if not isinstance(environments, list):
        environments = [environments] * len(tapes)

    def worker_func(
        input: tuple[TapeType, Environment], agent: Agent, max_loops: int, strict: bool
    ) -> TapeType | Exception:
        start_tape, env = input
        try:
            result = main_loop(agent, start_tape, env, max_loops=max_loops).get_final_tape()
        except Exception as e:
            if is_debug_mode() or strict:
                return e
            return start_tape.model_copy(
                update=dict(metadata=TapeMetadata(parent_id=start_tape.metadata.id, error=traceback.format_exc()))
            )
        result.metadata.parent_id = start_tape.metadata.id
        return result

    processor = choose_processor(n_workers=n_workers)
    worker_func_partial = partial(worker_func, agent=agent, max_loops=max_loops, strict=strict)
    for smth in processor(zip(tapes, environments), worker_func_partial):
        if isinstance(smth, Tape):
            yield smth
        else:
            raise smth


class ObsLayerConfig(BaseModel, Generic[TapeType, ObservationMakerTapeType]):
    obs_makers: list[tuple[ObservationMaker[TapeType, ObservationMakerTapeType], int]]


def batch_add_observations(
    tapes: list[TapeType],
    layer_conf: ObsLayerConfig[TapeType, ObservationMakerTapeType],
    n_workers: int = _DEFAULT_N_WORKERS,
    seed: int = 1,
    strict: bool = False,
    shuffle: bool = True,
) -> Generator[tuple[TapeType, ObservationMakerTapeType], None, None]:
    # TODO: add environments, use main loop
    def worker_func(input_: tuple[TapeType, ObservationMaker]) -> tuple[TapeType, ObservationMakerTapeType] | Exception:
        try:
            tape, obs_maker = input_
            for event in obs_maker.run(obs_maker.make_own_tape(tape)):
                if event.final_tape:
                    return obs_maker.add_observation(tape, event.final_tape), event.final_tape
        except Exception as e:
            logger.error(e, exc_info=True)
            if is_debug_mode() or strict:
                return e
            result_tape = tape.model_copy(update=dict(metadata=TapeMetadata(parent_id=tape.metadata.id, error=repr(e))))
            result_obs_maker_tape = obs_maker.make_own_tape(tape)
            result_obs_maker_tape.metadata.error = str(e)
            return result_tape, result_obs_maker_tape
        raise Exception("Observation maker did not finish the tape")

    # TODO: generate for many obs_makers at the same time
    rng = random.Random(seed)
    for i, (obs_maker, count) in enumerate(layer_conf.obs_makers):
        possible_tapes = [t for t in tapes if obs_maker.can_continue(t)]
        # TODO: what if len(possible_tapes) == 0?
        if shuffle:
            # take 'count' random tapes from possible_tapes using seed
            chosen_tapes = rng.choices(possible_tapes, k=count)
        else:
            # TODO: not fully deterministic because of obs_maker.can_continue()
            chosen_tapes = [possible_tapes[i % len(possible_tapes)] for i in range(count)]

        processor = choose_processor(n_workers=n_workers)
        for smth in processor(zip(chosen_tapes, [obs_maker] * count), worker_func):
            if isinstance(smth, tuple):
                yield smth
            else:
                raise smth


def batch_annotate(
    tapes: list[TapeType],
    annotator: Annotator[TapeType, AnnotatorTapeType],
    n_workers: int = _DEFAULT_N_WORKERS,
    strict: bool = False,
) -> Generator[AnnotatorTapeType, None, None]:
    def worker_func(input_: tuple[TapeType, Annotator]) -> AnnotatorTapeType | Exception:
        try:
            tape, annotator = input_
            return annotator.annotate(tape)
        except Exception as e:
            if is_debug_mode() or strict:
                return e
            result = annotator.make_own_tape(tape)
            result.metadata.error = str(e)
            return result

    processor = choose_processor(n_workers=n_workers)
    for smth in processor(zip(tapes, [annotator] * len(tapes)), worker_func):
        if isinstance(smth, Tape):
            yield smth
        else:
            raise smth


def generate_tapes(
    agent: Agent[TapeType],
    start_tapes: list[TapeType],
    environments: Environment | list[Environment],
    layers_conf: Sequence[ObsLayerConfig[TapeType, ObservationMakerTapeType]],
    work_dir: Path,
    annotator: Annotator[TapeType, AnnotatorTapeType] | None = None,
    n_workers: int = _DEFAULT_N_WORKERS,
    seed: int = 1,
    strict: bool = False,
    shuffle: bool = True,
) -> list[list[TapeType]]:
    tapes = start_tapes
    layer_tapes = [[]] * (len(layers_conf) * 2)

    for i, layer_conf in enumerate(layers_conf):
        begin_tapes_path = work_dir / f"tapes_{i}_begin.yaml"
        end_tapes_path = work_dir / f"tapes_{i}_end.yaml"
        obs_maker_tapes_path = work_dir / f"obs_maker_tapes_{i}.yaml"
        annotator_tapes_path = work_dir / f"annotator_tapes_{i}.yaml"
        logger.info(f"tapes: {len(tapes)}")

        # make observations
        new_layer_begin = []
        with (
            stream_yaml_tapes(begin_tapes_path) as tapes_dumper,
            stream_yaml_tapes(obs_maker_tapes_path) as obs_maker_tapes_dumper,
        ):
            for tape, obs_maker_tape in batch_add_observations(
                tapes, layer_conf, n_workers=n_workers, seed=seed, strict=strict, shuffle=shuffle
            ):
                tapes_dumper.save(tape)
                obs_maker_tapes_dumper.save(obs_maker_tape)
                if tape.metadata.error is None:
                    new_layer_begin.append(tape)
        logger.info(f"new_layer_begin: {len(new_layer_begin)} tapes")

        # run the agent
        new_layer_end = []
        with stream_yaml_tapes(end_tapes_path) as dumper:
            for tape in batch_main_loop(agent, new_layer_begin, environments, n_workers=n_workers, strict=strict):
                dumper.save(tape)
                if tape.metadata.error is None:
                    new_layer_end.append(tape)

        # run the annotator
        with stream_yaml_tapes(annotator_tapes_path) as dumper:
            if annotator is not None:
                for tape in batch_annotate(new_layer_end, annotator, n_workers=n_workers, strict=strict):
                    dumper.save(tape)

        logger.info(f"new_layer_end: {len(new_layer_end)} tapes")
        tapes = new_layer_end
        layer_tapes[2 * i] = new_layer_begin
        layer_tapes[2 * i + 1] = new_layer_end

    return layer_tapes

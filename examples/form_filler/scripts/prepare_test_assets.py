import os
import random
import sys
from pathlib import Path

import hydra
import yaml
from hydra.core.global_hydra import GlobalHydra
from hydra.utils import instantiate
from omegaconf import OmegaConf
from tqdm import tqdm

from ..tape import FormFillerTape
from ..user_simulator_agent import UserSimulatorTape
from .run_formfiller_agent import run_formfiller_agent

assets_folder = Path(__file__).parent.parent.parent.parent / "tests" / "examples" / "res" / "form_filler"

input_tapes_for_teacher_path = assets_folder / "input_tapes_for_teacher.yaml"
input_tapes_for_user_path = assets_folder / "input_tapes_for_user.yaml"
output_tapes_for_teacher_path = assets_folder / "output_tapes_for_teacher.yaml"
output_tapes_for_user_path = assets_folder / "output_tapes_for_user.yaml"


def validate_and_save_agent_configs():
    GlobalHydra.instance().clear()
    hydra.initialize(config_path="../conf", version_base="1.2")

    teacher_agent_config = hydra.compose(
        overrides=[
            "+agent=teacher_agent",
            "llm@agent.llm=openrouter_llama3_405b_temp0",
        ],
    )
    print("Attempting to initialize teacher agent")
    instantiate(teacher_agent_config.agent)
    print("-> success")

    print("Exporting teacher agent config")
    assets_folder.mkdir(exist_ok=True)
    with open(assets_folder / "teacher_agent_test_config.yaml", "w") as f:
        OmegaConf.save(teacher_agent_config, f)
    print(f'-> saved to {assets_folder / "teacher_agent_test_config.yaml"}')

    user_agent_config = hydra.compose(
        overrides=[
            "+user_simulator_agent=test_behavior",
            "+llm@user_simulator_agent.llms=openrouter_llama3_405b_temp0",
        ],
    )
    print("Attempting to initialize user simulator agent")
    instantiate(user_agent_config.user_simulator_agent)
    print("-> success")

    print("Exporting user simulator agent config")
    with open(assets_folder / "user_simulator_agent_test_config.yaml", "w") as f:
        OmegaConf.save(user_agent_config, f)
    print(f'-> saved to {assets_folder / "user_simulator_agent_test_config.yaml"}')

    # Now make some completions


def get_teacher_agent():
    # Reload from the saved config
    cfg = OmegaConf.load(assets_folder / "teacher_agent_test_config.yaml")
    return instantiate(cfg.agent)


def get_user_simulator_agent():
    # Reload from the saved config
    cfg = OmegaConf.load(assets_folder / "user_simulator_agent_test_config.yaml")
    return instantiate(cfg.user_simulator_agent)


def extract_random_tapes_from_tape_tree(
    tape_tree_path: Path,
    teacher_layers: tuple[int, ...] = (0, 2, 4, 8),
    user_layers: tuple[int, ...] = (0, 2, 4, 8),
    n_tapes_per_layer: int = 1,
):
    tape_tree_path = Path(tape_tree_path)

    random.seed(0)

    # Get input tapes for agent
    tapes_for_agent = []
    for layer in teacher_layers:
        print(f"Teacher Agent Inputs: Extracting {n_tapes_per_layer} tapes from layer {layer}")
        tape_tree_layer = tape_tree_path / f"layer_{layer}"
        with open(tape_tree_layer / "data.yaml") as f:
            tapes_obj = list(yaml.safe_load_all(f))
        random.shuffle(tapes_obj)
        tapes_obj = tapes_obj[:n_tapes_per_layer]
        for idx, tape_obj in enumerate(tapes_obj):
            tape = FormFillerTape.model_validate(tape_obj)
            tapes_for_agent.append(tape)

    # Get input tapes for user
    tapes_for_user = []
    for layer in user_layers:
        print(f"User Agent Inputs: Extracting {n_tapes_per_layer} tapes from layer {layer}")
        tape_tree_layer = tape_tree_path / f"layer_{layer}"
        with open(tape_tree_layer / "user_simulator_tapes.yaml") as f:
            tapes_obj = list(yaml.safe_load_all(f))
        random.shuffle(tapes_obj)
        tapes_obj = tapes_obj[:n_tapes_per_layer]
        for idx, tape_obj in enumerate(tapes_obj):
            tape = UserSimulatorTape.model_validate(tape_obj)
            # remove all the predicted steps
            tape.steps = []
            tape.metadata.n_added_steps = 0

            tapes_for_user.append(tape)

    # Save the tapes
    input_tapes_for_teacher_path.parent.mkdir(exist_ok=True, parents=True)
    input_tapes_for_user_path.parent.mkdir(exist_ok=True, parents=True)
    with open(input_tapes_for_teacher_path, "w") as f:
        yaml.safe_dump_all([tape.model_dump() for tape in tapes_for_agent], f)
    with open(input_tapes_for_user_path, "w") as f:
        yaml.safe_dump_all([tape.model_dump() for tape in tapes_for_user], f)
    print(f"Saved input tapes successfully to {input_tapes_for_teacher_path} and {input_tapes_for_user_path}")


def load_teacher_input_tapes() -> list[FormFillerTape]:
    with open(input_tapes_for_teacher_path) as f:
        tapes = list(yaml.safe_load_all(f))
    return [FormFillerTape.model_validate(tape) for tape in tapes]


def load_user_input_tapes() -> list[UserSimulatorTape]:
    with open(input_tapes_for_user_path) as f:
        tapes = list(yaml.safe_load_all(f))
    return [UserSimulatorTape.model_validate(tape) for tape in tapes]


def load_teacher_reference_tapes() -> list[FormFillerTape]:
    with open(output_tapes_for_teacher_path) as f:
        tapes = list(yaml.safe_load_all(f))
    return [FormFillerTape.model_validate(tape) for tape in tapes]


def load_user_reference_tapes() -> list[UserSimulatorTape]:
    with open(output_tapes_for_user_path) as f:
        tapes = list(yaml.safe_load_all(f))
    return [UserSimulatorTape.model_validate(tape) for tape in tapes]


def get_completions(save_as_references: bool = True):
    teacher_agent = get_teacher_agent()
    teacher_input_tapes = load_teacher_input_tapes()
    teacher_output_tapes = []
    print("Generating completions for teacher")
    for input_tape in tqdm(teacher_input_tapes):
        __, predicted_tape_or_exception = run_formfiller_agent(input_tape, teacher_agent)
        assert not isinstance(
            predicted_tape_or_exception, Exception
        ), f"Failed on input tape {input_tape.metadata.id}: {predicted_tape_or_exception}"
        teacher_output_tapes.append(predicted_tape_or_exception)
    print(f"-> successfully generated {len(teacher_output_tapes)} reference completions")
    if save_as_references:
        with open(output_tapes_for_teacher_path, "w") as f:
            yaml.safe_dump_all([tape.model_dump() for tape in teacher_output_tapes], f)

    user_agent = get_user_simulator_agent()
    user_input_tapes = load_user_input_tapes()
    user_output_tapes = []
    failed_user_inputs = []
    print("Generating completions for user")
    for input_tape in tqdm(user_input_tapes):
        try:
            output_tape = user_agent.run(input_tape).get_final_tape()
            user_output_tapes.append(output_tape)
        except Exception as e:
            print(f"Failed on input tape {input_tape.metadata.id}: {e}")
            failed_user_inputs.append(input_tape)
            continue
    print(f"-> successfully generated {len(user_output_tapes)} reference completions")
    if save_as_references:
        with open(output_tapes_for_user_path, "w") as f:
            yaml.safe_dump_all([tape.model_dump() for tape in user_output_tapes], f)
    assert len(user_input_tapes) == len(
        user_output_tapes
    ), f"-> failed user input ids: {[tape.metadata.id for tape in failed_user_inputs]}"
    return teacher_output_tapes, user_output_tapes


def assert_same_tape(tape, tape_ref):
    assert tape == tape_ref, f"Tapes {tape.metadata.id} and {tape_ref.metadata.id} do not match"
    # assert len(tape.steps) == len(tape_ref.steps), f'Tapes {tape.metadata.id} and {tape_ref.metadata.id} have different number of steps'
    # for idx, (step, step_ref) in zip(tape.steps, tape_ref.steps):
    #     assert step == step_ref, f'Step {idx} of tapes {tape.metadata.id} and {tape_ref.metadata.id} do not match'


def predict_and_compare():
    # Make new predictions
    teacher_output_tapes, user_output_tapes = get_completions(save_as_references=False)
    teacher_reference_tapes = load_teacher_reference_tapes()
    user_reference_tapes = load_user_reference_tapes()

    # Compare the new predictions with the reference predictions
    assert len(teacher_output_tapes) == len(teacher_reference_tapes), "Number of teacher output tapes do not match"
    assert len(user_output_tapes) == len(user_reference_tapes), "Number of user output tapes do not match"

    # Check that the predictions are the same
    for tape, tape_ref in zip(teacher_output_tapes, teacher_reference_tapes):
        assert_same_tape(tape, tape_ref)


def prepare_test_assets(tape_tree_path: str):
    # # # This reads hydra configs from the examples/form_filler/conf directory
    validate_and_save_agent_configs()

    # # # Extract some tapes randomly from an existing dialogue tree created by make_tape_tree
    tape_tree_path = Path(tape_tree_path)
    extract_random_tapes_from_tape_tree(
        tape_tree_path,
        teacher_layers=(0, 2, 4),
        user_layers=(0, 2, 4),
        n_tapes_per_layer=1,
    )

    # Generate the reference completions
    # If any of the input tapes cause agent failure, you may manually remove them from the input tapes yaml and rerun the script
    # make sure to keep extract_random_tapes_from_tape_tree() commented out to avoid regenerating the same tapes

    # Patch environment variable so that tapedata.sqlite is saved in the right place
    os.environ["TAPEAGENTS_SQLITE_DB"] = os.path.join(assets_folder, "tapedata.sqlite")
    os.remove(os.environ["TAPEAGENTS_SQLITE_DB"]) if os.path.exists(
        os.environ["TAPEAGENTS_SQLITE_DB"]
    ) else None  # clean up database
    get_completions(save_as_references=True)


if __name__ == "__main__":
    # Run this script as python -m examples.form_filler.scripts.prepare_test_assets <path_to_tape_tree_dir>
    # We do NOT make use of make_test_data because
    # Hydra configuration clashes with make_test_data.py's directory changes (hydra.initialize does not support absolute paths)
    prepare_test_assets(sys.argv[1])

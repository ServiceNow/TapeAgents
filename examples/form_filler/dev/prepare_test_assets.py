import os
from pathlib import Path

assets_folder = Path(__file__).parent.parent.parent.parent / 'tests' / 'res' / 'form_filler'
# Set up LLM caching sqlite db path (test will reuse this instead of calling LLM)
os.environ["TAPEAGENTS_SQLITE_DB"] = os.path.join(assets_folder, "tapedata.sqlite")


input_tapes_for_teacher_path = assets_folder / 'input_tapes_for_teacher.yaml'
input_tapes_for_user_path = assets_folder / 'input_tapes_for_user.yaml'
output_tapes_for_teacher_path = assets_folder / 'output_tapes_for_teacher.yaml'
output_tapes_for_user_path = assets_folder / 'output_tapes_for_user.yaml'



import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
from hydra.core.global_hydra import GlobalHydra
from tqdm import tqdm
import yaml
import random

from examples.form_filler.dev.run_formfiller_agent import run_formfiller_agent
from examples.form_filler.dev.run_user_simulator import run_user_simulator_agent
from examples.form_filler.tape import FormFillerTape






def validate_and_save_agent_configs():
    GlobalHydra.instance().clear()
    hydra.initialize(config_path="../conf", version_base="1.2")

    teacher_agent_config = hydra.compose(
        overrides=[
            f"+agent=teacher_agent",
            f"llm@agent.llm=vllm_llama3_405b_temp1",
        ],
    )
    print('Attempting to initialize teacher agent')
    agent = instantiate(teacher_agent_config.agent)
    print('-> success')

    print('Exporting teacher agent config')
    assets_folder.mkdir(exist_ok=True)
    with open(assets_folder / 'teacher_agent_test_config.yaml', 'w') as f:
        OmegaConf.save(teacher_agent_config, f)
    print(f'-> saved to {assets_folder / "teacher_agent_test_config.yaml"}')

    user_agent_config = hydra.compose(
        overrides=[
            f"+user_simulator_agent=test_behavior",
            f"+llm@user_simulator_agent.llms=vllm_llama3_405b_temp1"
        ],
    )
    print('Attempting to initialize user simulator agent')
    user_agent = instantiate(user_agent_config.user_simulator_agent)
    print('-> success')

    print('Exporting user simulator agent config')
    with open(assets_folder / 'user_simulator_agent_test_config.yaml', 'w') as f:
        OmegaConf.save(user_agent_config, f)
    print(f'-> saved to {assets_folder / "user_simulator_agent_test_config.yaml"}')

    # Now make some completions


def get_teacher_agent():
    # Reload from the saved config
    cfg = OmegaConf.load(assets_folder / 'teacher_agent_test_config.yaml')
    return instantiate(cfg.agent)

def get_user_simulator_agent():
    # Reload from the saved config
    cfg = OmegaConf.load(assets_folder / 'user_simulator_agent_test_config.yaml')
    return instantiate(cfg.user_simulator_agent)

def extract_random_tapes_from_tape_tree(tape_tree_path: Path, layers_to_extract: tuple[int] = (0, 2, 4, 8), n_tapes_per_layer: int = 3):
    tape_tree_path = Path(tape_tree_path)

    random.seed(0)

    # Get input tapes for agent
    tapes_for_agent = []
    for layer in [0, 2, 4, 8]:
        print(f'Teacher Agent Inputs: Extracting {n_tapes_per_layer} tapes from layer {layer}')
        tape_tree_layer = tape_tree_path / f'layer_{layer}'
        with open(tape_tree_layer / 'data.yaml') as f:
            tapes_obj = list(yaml.safe_load_all(f))
        random.shuffle(tapes_obj)
        for idx, tape_obj in enumerate(tapes_obj):
            tape = FormFillerTape.model_validate(tape_obj)
            tapes_for_agent.append(tape)
            if idx >= 3:
                break
        
    # Get input tapes for user
    tapes_for_user = []
    for layer in [1, 3, 5, 9]:
        print(f'User Agent Inputs: Extracting {n_tapes_per_layer} tapes from layer {layer}')
        tape_tree_layer = tape_tree_path / f'layer_{layer}'
        with open(tape_tree_layer / 'data.yaml') as f:
            tapes_obj = list(yaml.safe_load_all(f))
        random.shuffle(tapes_obj)
        for idx, tape_obj in enumerate(tapes_obj):
            tape = FormFillerTape.model_validate(tape_obj)
            tapes_for_user.append(tape)
            if idx >= 3:
                break

    # Save the tapes
    input_tapes_for_teacher_path.parent.mkdir(exist_ok=True, parents=True)
    input_tapes_for_user_path.parent.mkdir(exist_ok=True, parents=True)
    with open(input_tapes_for_teacher_path, 'w') as f:
        yaml.safe_dump_all([tape.model_dump() for tape in tapes_for_agent], f)
    with open(input_tapes_for_user_path, 'w') as f:
        yaml.safe_dump_all([tape.model_dump() for tape in tapes_for_user], f)
    print(f'Saved input tapes successfully to {input_tapes_for_teacher_path} and {input_tapes_for_user_path}')

def load_teacher_input_tapes() -> list[FormFillerTape]:
    with open(input_tapes_for_teacher_path) as f:
        tapes = list(yaml.safe_load_all(f))
    return [FormFillerTape.model_validate(tape) for tape in tapes]


def load_user_input_tapes() -> list[FormFillerTape]:
    with open(input_tapes_for_user_path) as f:
        tapes = list(yaml.safe_load_all(f))
    return [FormFillerTape.model_validate(tape) for tape in tapes]


def prepare_reference_completions():
    # teacher_agent = get_teacher_agent()
    # teacher_input_tapes = load_teacher_input_tapes()
    # teacher_output_tapes = []
    # print('Generating reference completions for teacher')
    # for input_tape in tqdm(teacher_input_tapes):
    #     __, predicted_tape_or_exception = run_formfiller_agent(input_tape, teacher_agent)
    #     assert not isinstance(predicted_tape_or_exception, Exception), f'Failed on input tape {input_tape.metadata.id}'
    #     teacher_output_tapes.append(predicted_tape_or_exception)
    # print(f'-> successfully generated {len(teacher_output_tapes)} reference completions')
    # with open(output_tapes_for_teacher_path, 'w') as f:
    #     yaml.safe_dump_all([tape.model_dump() for tape in teacher_output_tapes], f)
    
    user_agent = get_user_simulator_agent()
    user_input_tapes = load_user_input_tapes()
    user_output_tapes = []
    failed_user_inputs = []
    print('Generating reference completions for user')
    for input_tape in tqdm(user_input_tapes):
        exception, continued_tape, user_simulator_agent_tape = run_user_simulator_agent(input_tape, user_agent)
        if exception:
            print(f'Failed on input tape {input_tape.metadata.id}')
            failed_user_inputs.append(input_tape)
            continue
        user_output_tapes.append(continued_tape)
    print(f'-> successfully generated {len(user_output_tapes)} reference completions')
    with open(output_tapes_for_user_path, 'w') as f:
        yaml.safe_dump_all([tape.model_dump() for tape in user_output_tapes], f)
    assert len(user_input_tapes) == len(user_output_tapes), (f'-> failed user input ids: {[tape.metadata.id for tape in failed_user_inputs]}')





def main():
    # # Run this script as python -m examples.form_filler.dev.prepare_test_assets
    validate_and_save_agent_configs()

    # Extract some tapes randomly from an existing dialogue tree created by make_tape_tree
    # extract_random_tapes_from_tape_tree('/mnt/llmd/data/gabriel/make_tape_tree/train/FlyCorp/agent_teacher_agent_vllm_llama3_405b_temp1/user_vllm_llama3_405b_temp1/tree_config6_size500/dec2')

    # Generate the reference completions
    # If any of the input tapes cause agent failure, you may manually remove them from the input tapes yaml and rerun the script
    # make sure to keep extract_random_tapes_from_tape_tree() commented out to avoid regenerating the same tapes
    prepare_reference_completions()




if __name__ == "__main__":
    main()
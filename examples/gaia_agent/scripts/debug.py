import os
from pprint import pprint

from tapeagents.io import save_json_tape
from tapeagents.llms import LiteLLM
from tapeagents.observe import retrieve_llm_call
from tapeagents.orchestrator import main_loop

from ..environment import GaiaEnvironment
from ..eval import load_dataset, task_to_question_step
from ..tape import GaiaTape
from ..v2 import GaiaPlanner


def main():
    dset = load_dataset("../gaia/dataset/validation/")
    exp_dir = "../gaia/runs/v2_debug/"
    tapes_dir = f"{exp_dir}/tapes"
    os.environ["TAPEAGENTS_SQLITE_DB"] = os.path.join(exp_dir, "tapedata.sqlite")
    tape_name = "debug"
    tasks = dset[2]
    task = tasks[0]
    pprint(task, width=140)
    llm = LiteLLM(model_name="gpt-4o-mini-2024-07-18", context_size=128000, parameters={"temperature": 0.2})
    env = GaiaEnvironment(vision_lm=llm, safe_calculator=False)
    planner = GaiaPlanner.create(llm)
    tape = GaiaTape(steps=[task_to_question_step(task, env)])
    metadata = tape.metadata
    metadata.task = task
    metadata.level = 2
    try:
        for event in main_loop(planner, tape, env, max_loops=30):
            if event.agent_event and event.agent_event.step:
                step = event.agent_event.step
                tape = tape.append(step)
                llm_call = retrieve_llm_call(step.metadata.prompt_id)
                assert llm_call
                print("PROMPT:")
                for i, m in enumerate(llm_call.prompt.messages):
                    print(f"M{i+1}")
                    pprint(m, width=140)
                print(f"{len(tape)} STEP: {step.llm_view()}")
                print("=" * 80)
                input("Press Enter to continue...")
    finally:
        tape.metadata = metadata
        save_json_tape(tape, tapes_dir, tape_name)
        print(f"Saved tape to {tapes_dir}/{tape_name}.json")


if __name__ == "__main__":
    main()

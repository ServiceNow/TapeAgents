**Code that lives here is meant to be moved over to the real TapeAgent repository.**

# GREADTH Agent
The GREADTH Agent is an agent that fills a form and is great at it.
GREADTH stands for Grounded, REsponsive, Accurate, Disciplined, Transparent and Helpful.

## Requirements
- Make sure to activate the `tapeagents` conda environment.
- Install additional requirements for this example `pip install -r requirements.formfiller.txt`


## Form-filler Agent

### Teacher Agent
The teacher agent decomposes the conversational form-filling task into smaller reasoning nodes before each agent message.
The nodes are defined in [teacher.py](https://github.com/ServiceNow/cat-mono-repo/blob/main/llmd2-core/src/llmd2/tapeagents_tmp/ghreat/teacher.py) as 

```
 ---v
|  RoutingNode ____if-requested-call-confirmation______
|   v                                                  |
|  IntentDiscoveryNode <-|                             |
|   v                |___| until InspectFunction       |
|  GatherValuesNode                                    |
|   v                                                  |
|  VerifyValuesNode                                    |
|   v                                                  |
|  RetrospectivePlanNode                               |
|   v                                                  |
|  ForwardPlanNode_____if-requesting-call-confirmation |
|   v                      |                           |
|--GenerationNode          |                           |
|                          |                           |
|--WriteConfirmationNode <-|                           |
    ^                                                  |
   CallFunctionNode* <---------------------------------|
```

`GenerationNode` and `WriteConfirmationNode` yield action steps, thus will stop the agent.run() loop. If the agent gets called again it must restart from the `RoutingNode`.
`CallFunctionNode*` can either yield thoughts (in which case we go to `WriteConfirmationNode` to ask again for confirmation, and back to `RoutingNode`), or an action step (in which case the agent.run() loop and the overall dialog terminates).

The prompts for the nodes are provided in [teacher.yaml](https://github.com/ServiceNow/cat-mono-repo/blob/main/llmd2-core/src/llmd2/tapeagents_tmp/ghreat/conf/teacher.yaml).

The original implementations: the [prompter](https://github.com/ServiceNow/cat-mono-repo/blob/main/llmd2-core/src/llmd2/prompters/multistep_prompter.py) and the [config](https://github.com/ServiceNow/cat-mono-repo/blob/main/conf/prompter/multistep_nico_teacher_prompt_v5_chat.yaml)

To test the teacher, run the following command:

```bash
python tapeagents/tapeagents/examples/ghreat/dev/run_formfiller_agent.py \
    agent=teacher_agent \
    user_dialogues_path=/mnt/llmd/data/ehsan_kamalloo/tapes/make_balanced_set/ehsan_balanced_v3/balanced_ufs_part0_testset.yaml \
    output_path=.local/debug/agent_tapes \
    n_workers=2
```

This script will produce agent tapes stored in the `output_path` directory, which would be in the above command `.local/debug/agent_tapes/`. The structure of the files in the output directory is:
- `teacher_predicted_tapes.yaml`: Successful agent tapes
- `teacher_failure_tapes.yaml`: Failed agent tapes with an error step as the last step
- `teacher_stats.json`: The counts of last step types of failed and successful tapes

Note that `balanced_ufs_part0_testset.yaml` is a small subset with 8 tapes. The entire dev set is here: `/mnt/llmd/data/ehsan_kamalloo/tapes/make_balanced_set/ehsan_balanced_v3/balanced_ufs_part0.yaml`. The agent tapes on the entire dev set is stored here: `/mnt/llmd/data/ehsan_kamalloo/tapes/make_balanced_set/ehsan_balanced_v3/part0/`.

It takes a few hours to run the teacher agent on the entire file with no parallelization (`n_workers=0`) and takes around 20 minutes when `n_workers=6`.

### Student Agent
The student agent is designed for fine-tuning a small model, therefore it is much simpler.
The nodes are defined in [student.py](https://github.com/ServiceNow/cat-mono-repo/blob/main/llmd2-core/src/llmd2/tapeagents_tmp/ghreat/student.py) as 

```
CheckFunctionCandidatesNode
 |
IntentDiscoveryNode <-|
 |                |___| until InspectFunction
StudentNode <-|
          |___| until end of dialog
```

The prompts for the nodes are provided in `student.yaml` ([here](https://github.com/ServiceNow/cat-mono-repo/blob/main/llmd2-core/src/llmd2/tapeagents_tmp/ghreat/conf/student.yaml)). 

The original implementations: the [prompter](https://github.com/ServiceNow/cat-mono-repo/blob/main/llmd2-core/src/llmd2/prompters/gff_chat_prompter_v2.py) and the [config](https://github.com/ServiceNow/cat-mono-repo/blob/main/conf/prompter/gff_chat_prompter_v2.yaml)

To test the student, run the following command:

```bash
python tapeagents/tapeagents/examples/ghreat/dev/run_formfiller_agent.py \
    agent=student \
    user_dialogues_path=/mnt/llmd/data/ehsan_kamalloo/tapes/make_balanced_set/ehsan_balanced_v3/balanced_ufs_part0_testset.yaml \
    output_path=.local/debug/agent_tapes \
    n_workers=2
```

This script will produce agent tapes stored in the `output_path` directory, which would be in the above command `.local/debug/agent_tapes/`. The structure of the files in the output directory is:
- `student_predicted_tapes.yaml`: Successful agent tapes
- `student_failure_tapes.yaml`: Failed agent tapes with an error step as the last step
- `student_stats.json`: The counts of last step types of failed and successful tapes

### VISUALIZATION

To look at the generated tapes in the browser, run the following command:
```bash
python tapeagents/tapeagents/examples/ghreat/dev/visualize_formfiller_tapes.py <OUTPUT_PATH>
```
replace the `<OUTPUT_PATH>` with `.local/debug/agent_tapes` for the example above.

Then open your browser to http://localhost:7860/


## User Agent

The user agent is designed to simulate user interacting with the form-filler Agent.
The user agent takes as input a `FormFillerTape` that was edited last by the teacher or student agent, and adds a `UserStep` to it.

By design `UserStep` is an `Observation`, but Agents can only yield `Action` or `Thought` steps. `Observation` steps are yielded by the `Environment`.
As such, the user agent is a special case of pseudo-environment agent that must yield a `UserStep` as a response to the last `AssistantStep`.

This is done by yielding a special action called `MakeObservation` that wraps any `Observation`.
It is used in the `ObservationMaker` agent in TapeAgents (see [here](https://github.com/ServiceNow/TapeAgents/blob/main/tapeagents/agent.py)).
The user agent will create its own `UserSimulatorTape` that takes a `FormFillerTape` as context, and contains `MakeObservation` steps (among other user agent specific steps).
```python
# The user simulator tape wraps the formfiller tape as its context
# It produces some intermediate steps and a final MakeObservation step
UserSimulatorTape = Tape[FormFillerTape, ... | MakeObservation]
```

The user agent will then run `self.add_observation(form_filler_tape, user_tape)` to convert its `MakeObservation` steps to `Observation` steps and append them to the input `form_filler_tape`.
This code can be seen in [run_user_simulator.run_user_simulator_agent()](https://github.com/ServiceNow/cat-mono-repo/blob/main/llmd2-core/src/llmd2/tapeagents_tmp/ghreat/dev/run_user_simulator.py).

The user agent nodes are defined in [user_simulator_agent.py](https://github.com/ServiceNow/cat-mono-repo/blob/main/llmd2-core/src/llmd2/tapeagents_tmp/ghreat/user_simulator_agent.py) as 

```
SampleUserInstructionNode
 |
UserSimulatorMainNode
```

The main prompt for the user agent is defined in [base.yaml](https://github.com/ServiceNow/cat-mono-repo/blob/main/llmd2-core/src/llmd2/tapeagents_tmp/ghreat/conf/user_simulator_agent/base.yaml).

To test various user behaviors we defined various prompts for the user agent, all defined in [conf/user_simulator_agent/](https://github.com/ServiceNow/cat-mono-repo/blob/main/llmd2-core/src/llmd2/tapeagents_tmp/ghreat/conf/user_simulator_agent).

To test the user with the default "happy_path" behavior, run the following command:

```bash
USER_SIMULATOR="happy_path"  # or any of the behaviors defined in conf/user_simulator_agent/
python tapeagents/tapeagents/examples/ghreat/dev/run_user_simulator.py \
    input_dialogues_path=".local/debug/agent_tapes/teacher_predicted_tapes.yaml" \
    output_path=".local/debug/user_tapes/${USER_SIMULATOR}" \
    user_simulator_agent="${USER_SIMULATOR}" \
    max_continuable_tapes=30 \
    n_workers=4
```

This script will produce both *user tapes* and *form filler agent tapes* stored in the `output_path` directory
The structure of the files in the output directory is:
- `user_simulator_tapes.yaml`: all **user** tapes (of type `UserSimulatorTape`) generated by the user agent
- `user_predicted_tapes.yaml`: successfully continued agent tapes (of type `FormFillerTape`)
- `user_failed_tapes.yaml`: agent tapes that could not be continued
- `counters.json`: the number of tapes in each file


## Generating synthetic dialogues (tapes)

Synthetic data can be used for:
- Evaluating the teacher form-filler agent by simulating its interaction with a user.
- Distilling a teacher form-filler agent using a large LLM (complex and expensive) , into a "student" agent using a smaller LLM (simpler and cheaper).

We generate synthetic dialogues by alternating between calling the Form-filler Agent and User Agent to continue the dialogue tape.

For instance, to generate trees using the `tree_config6` configuration, 
run the following command (in a tmux window)
```bash
python llmd2-core/src/llmd2/tapeagents_tmp/ghreat/dev/make_tape_tree.py \
  make_tape_tree.force_restart_idx=-1 \
  make_tape_tree.output_path=<path to output folder> \
  make_tape_tree.preambles_path=<path to forms>/train/FlyCorp \
  make_tape_tree.global_count=500 \
  make_tape_tree.user_simulator_llm=vllm_llama3_405b_temp1 \
  make_tape_tree.num_workers=6 \
  agent@make_tape_tree.agent=teacher_agent \
  llm@make_tape_tree.agent.llm=vllm_llama3_405b_temp1
```
This will generate a 18-layer "dialogue tree" with a width of 500. 
User Agents are randomly sampled according to a distribution defined in 
You may change the user/agent LLM by overriding `llm@make_tape_tree.agent.llm` or `make_tape_tree.user_simulator_llm` (see Hydra documentation).
Use `force_restart_idx=-1` to automatically resume from the last layer, or a non-negative value to restart from layer `force_restart_idx` (this will discard all subsequent layers, including the current one).

The resulting folder structure is a follows:
```
./layer_0:
data.yaml  DONE  user_simulator_tapes.yaml

./layer_1:
data.yaml  DONE  failures.yaml

./layer_2:
data.yaml  DONE  failures.yaml  user_simulator_tapes.yaml

./layer_3:
data.yaml  DONE  failures.yaml
...
formfiller_agent_tapes.yaml
formfiller_agent_tape_failures.yaml
formfiller_user_tapes.yaml 
```

Each layer corresponds to a User or Form-filler Agent turn. Even layers (`layer_0, layer_2, ...`) end with Users turns, while Odd layers (`layer_1, layer_3, ...`) end with Form-Filler agent turns:
- `data.yaml` contains successful user/agent completions, stored as `FormFillerTape` tapes.
- `failures.yaml` contain the dialogues (tapes) that could not be continued, stored as `FormFillerTape`.
- Additionally, for even layers, `user_simulator_tapes` contains tapes that are internal to the User Simulator, stored as `UserSimulatorTape`,

For convenience, we merge all agent (resp. ) dialogues into a single file `formfiller_agent_tapes.yaml` (resp. `formfiller_user_tapes.yaml`)

You can visualize the generated dialogues by running
```
python tapeagents/tapeagents/examples/ghreat/dev/visualize_formfiller_tapes.py <path to output folder>/layer_idx <optional: give path to tapedata.sqlite to visualize prompts>
```


## Critic Agent
The Critic agent is an agent that evaluates a GHREAT Agent tape according to the following 6 metrics:
- Grounded: does the agent lie?
- Helpful: does the agent move the conversation forward?
- Responsive: does the agent address all user messages?
- Accurate: does the agent correctly fill the form based on user messages?
- Transparent 1: does the agent follow his chain of thought?
- Transparent 2: does the agent mention all the notes it takes when it fills the form.

## Quickstart (Evaluation of the Critic Agent)
We have a set of ~1000 tapes evaluated by Toloka here: `/mnt/llmd/data/gontiern/tapes/CoffeeCorpGoldV2_annotations.yaml`
This set has ~100 tapes with "Yes" labels and ~100 tapes with "No" labels for each of the 6 metrics.

A balanced subset ((5 yes + 5 no) * 6 = 60 tapes) of this data can be found here: `/mnt/llmd/data/gontiern/tapes/tmp_annotations.yaml`

To test the Critic Agent against either of these, run the following command:
```bash
python tapeagents/tapeagents/examples/ghreat/dev/test_critique.py
    groundtruth_tapes_path=/path/to/golden_tapefile.yaml
    predicted_tapes_path=/path/to/predicted_tapefile.yaml
    n_workers=6
```

To run the Critic Agent on 1 tape in a web interface, run the following command:
```bash
python tapeagents/tapeagents/examples/ghreat/scripts/develop.py agent=critic
```

## Results
Running `test_critique.py` will result in the following:
- the set of annotated tapes in the location specified by `{predicted_tapes_path}` (eg: `/mnt/llmd/data/gontiern/tapes/CoffeeCorpGoldV2_annotations_predicted.yaml`)
- the scores for each metric in a file called `{predicted_tapes_path}_metrics.json` (eg: `/mnt/llmd/data/gontiern/tapes/CoffeeCorpGoldV2_annotations_predicted_metrics.json`)

Metrics on `CoffeeCorpGoldV2_annotations.yaml`:
```json
{
  "helpful": {
    "total": 200,
    "counts": 152,
    "score": 0.76
  },
  "responsive": {
    "total": 199,
    "counts": 131,
    "score": 0.6582914572864321
  },
  "transparent1": {
    "total": 184,
    "counts": 146,
    "score": 0.7934782608695652
  },
  "transparent2": {
    "total": 108,
    "counts": 88,
    "score": 0.8148148148148148
  },
  "grounded": {
    "total": 200,
    "counts": 140,
    "score": 0.7
  },
  "accurate": {
    "total": 200,
    "counts": 151,
    "score": 0.755
  }
}
```

Metrics on `tmp_annotations.yaml`
```json
{
  "transparent1": {
    "total": 10,
    "counts": 7,
    "score": 0.7
  },
  "transparent2": {
    "total": 10,
    "counts": 7,
    "score": 0.7
  },
  "grounded": {
    "total": 10,
    "counts": 8,
    "score": 0.8
  },
  "accurate": {
    "total": 10,
    "counts": 9,
    "score": 0.9
  },
  "responsive": {
    "total": 10,
    "counts": 8,
    "score": 0.8
  },
  "helpful": {
    "total": 10,
    "counts": 7,
    "score": 0.7
  }
}
```
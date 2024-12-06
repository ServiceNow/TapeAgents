# GREADTH Agent
The GREADTH Agent is an agent that fills a form and is great at it.
GREADTH stands for Grounded, REsponsive, Accurate, Disciplined, Transparent and Helpful.

## Requirements
- Make sure to activate the `tapeagents` conda environment.
- Install additional requirements for this example `pip install -r requirements.formfiller.txt`


## Form-filler Agent

### Teacher Agent
The teacher agent decomposes the conversational form-filling task into smaller reasoning nodes before each agent message.
The nodes are defined in [teacher.py](https://github.com/ServiceNow/TapeAgents/blob/formfiller/examples/form_filler/teacher.py) as 

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

The prompts for the nodes are provided in [teacher_agent.yaml](https://github.com/ServiceNow/TapeAgents/blob/formfiller/examples/form_filler/conf/agent/teacher_agent.yaml).

To test the teacher you need to have user tapes as input.
Assuming you already ran a [User Agent](#user-agent) with behavior `<B>`, you can run the following command to launch the teacher agent on the user tapes:
```bash
python -m examples.form_filler.dev.run_formfiller_agent \
    agent=teacher_agent \
    user_dialogues_path=.local/user_tapes/<B>/user_predicted_tapes.yaml \
    output_path=.local/agent_tapes/teacher
```
reaplcing <B> with the actual behavior name you used to run a user agent.

By default this script will use the [openrouter_llama3_405b_temp1](https://github.com/ServiceNow/TapeAgents/blob/formfiller/examples/form_filler/conf/llm/openrouter_llama3_405b_temp1.yaml) llm config which uses openrouter to call llama405B.
To use this model you must have an openrouter API key saved into your `TAPEAGENTS_LLM_TOKEN` environment variable.
You can check the pricing of this model on the [openrouter website](https://openrouter.ai/meta-llama/llama-3.1-405b-instruct).
To override the llm config used, specify `llm@agent.llm=...` in your command.

This script will produce agent tapes (1 per input user tape) stored in the `output_path` directory. The structure of the files in the output directory is:
- `teacher_agent_predicted_tapes.yaml`: Successful agent tapes
- `teacher_agent_failure_tapes.yaml`: Failed agent tapes with an error step as the last step
- `teacher_agent_stats.json`: The counts of last step types of failed and successful tapes

### Student Agent
The student agent is designed for a small fine-tuned model, therefore it is much simpler.
The nodes are defined in [student.py](https://github.com/ServiceNow/TapeAgents/blob/formfiller/examples/form_filler/student.py) as 

```
CheckFunctionCandidatesNode
 |
IntentDiscoveryNode <-|
 |                |___| until InspectFunction
StudentNode <-|
          |___| until end of dialog
```

The prompts for the nodes are provided in [student_agent.yaml](https://github.com/ServiceNow/TapeAgents/blob/formfiller/examples/form_filler/conf/agent/student_agent.yaml). 

To test the student you need to have user tapes as input.
Assuming you already ran a [User Agent](#user-agent) with the behavior `<B>`, you can run the following command to launch the student agent on the user tapes:
```bash
python -m examples.form_filler.dev.run_formfiller_agent \
    agent=student_agent \
    user_dialogues_path=.local/user_tapes/<B>/user_predicted_tapes.yaml \
    output_path=.local/agent_tapes/student
```
reaplcing <B> with the actual behavior name you used to run a user agent.

By default this script will use the [sft_llama3_8b_temp1](https://github.com/ServiceNow/TapeAgents/blob/formfiller/examples/form_filler/conf/llm/sft_llama3_8b_temp1.yaml) llm config.
Right now this config uses openrouter to call llama8B, however this config is meant to be used with a **fine-tuned** model. You should change the config `model_name` and `base_url` to map to your pretrained model.
Otherwise, to use this model you must have an openrouter API key saved into your `TAPEAGENTS_LLM_TOKEN` environment variable.
You can check the pricing of this model on the [openrouter website](https://openrouter.ai/meta-llama/llama-3.1-8b-instruct)
To override the llm config used, specify `llm@agent.llm=...` in your command.

This script will produce agent tapes stored in the `output_path` directory. The structure of the files in the output directory is:
- `student_agent_predicted_tapes.yaml`: Successful agent tapes
- `student_agent_failure_tapes.yaml`: Failed agent tapes with an error step as the last step
- `student_agent_stats.json`: The counts of last step types of failed and successful tapes

### VISUALIZATION

To look at the generated tapes in the browser, run the following command:
```bash
python -m examples.form_filler.dev.visualize_formfiller_tapes .local/agent_tapes/<teacher or student>
```

Then open your browser to http://localhost:7680


## User Agent

The user agent is designed to simulate user interacting with the form-filler Agent.
The user agent takes as input a `FormFillerTape` that was edited last by the teacher or student agent, and adds a `UserStep` to it.

By design `UserStep` is an `Observation`, but Agents can only yield `Action` or `Thought` steps. `Observation` steps are yielded by the `Environment`.
As such, the user agent is a special case of pseudo-environment agent that must yield a `UserStep` as a response to the last `AssistantStep`.

This is done by yielding a special action called `MakeObservation` that wraps any `Observation` (see the `ObservationMaker` class in [agent.py](https://github.com/ServiceNow/TapeAgents/blob/formfiller/tapeagents/agent.py)).
The user agent will create its own `UserSimulatorTape` that takes a `FormFillerTape` as context, and contains `MakeObservation` steps (among other user agent specific steps).
```python
# The user simulator tape wraps the formfiller tape as its context
# It produces some intermediate steps and a final MakeObservation step
UserSimulatorTape = Tape[FormFillerTape, ... | MakeObservation]
```

The user agent will then run `self.add_observation(form_filler_tape, user_tape)` to convert its `MakeObservation` steps to `Observation` steps and append them to the input `form_filler_tape`.
This code can be seen in the `run_user_simulator_agent()` function of [run_user_simulator.py](https://github.com/ServiceNow/TapeAgents/blob/formfiller/examples/form_filler/dev/run_user_simulator.py).

The user agent nodes are defined in [user_simulator_agent.py](https://github.com/ServiceNow/TapeAgents/blob/formfiller/examples/form_filler/user_simulator_agent.py) as 

```
SampleUserInstructionNode
 |
UserSimulatorMainNode
```

The main prompt for the user agent is defined in [base.yaml](https://github.com/ServiceNow/TapeAgents/blob/formfiller/examples/form_filler/conf/user_simulator_agent/base.yaml).
To test various user behaviors we defined various prompts for the user agent, all defined in [conf/user_simulator_agent/](https://github.com/ServiceNow/TapeAgents/blob/formfiller/examples/form_filler/conf/user_simulator_agent).

To test the user you need to have agent tapes as input. We provide a default starting agent tape in [initial_agent_tapes.yaml](https://github.com/ServiceNow/TapeAgents/blob/formfiller/examples/form_filler/initial_agent_tapes.yaml).
The first time you run the user use this starting tape as input:
```bash
USER_SIMULATOR="init_message_short"  # use init_message_... behaviors on begining of conversations and any other behaviors defined in conf/user_simulator_agent/ once the user specified its intent.
python -m examples.form_filler.dev.run_user_simulator \
    input_dialogues_path="examples/form_filler/initial_agent_tapes.yaml" \
    output_path=".local/user_tapes/${USER_SIMULATOR}" \
    user_simulator_agent="${USER_SIMULATOR}" \
    max_continuable_tapes=5  # specify the number of user tapes you want (maximum is the number of tapes in input_dialogues_path)
```

Otherwise, assuming you already ran a Teacher [Agent](#teacher-agent) in `.local/agent_tapes/teacher`, you can set `input_dialogues_path=".local/agent_tapes/teacher/teacher_predicted_tapes.yaml" \` to launch the user agent on the teacher agent tapes.

By default this script will use the [openrouter_llama3_405b_temp1](https://github.com/ServiceNow/TapeAgents/blob/formfiller/examples/form_filler/conf/llm/openrouter_llama3_405b_temp1.yaml) llm config which uses openrouter to call llama405B.
To use this model you must have an openrouter API key saved into your `TAPEAGENTS_LLM_TOKEN` environment variable.
You can check the pricing of this model on the [openrouter website](https://openrouter.ai/meta-llama/llama-3.1-405b-instruct).
To override the llm config used, specify `llm@user_simulator_agent.llms=...` in your command.

This script will produce 5 *user tapes* and 5 *form filler agent tapes* stored in the `output_path` directory
The structure of the files in the output directory is:
- `user_simulator_tapes.yaml`: all **user** tapes (of type `UserSimulatorTape`) generated by the user agent
- `user_predicted_tapes.yaml`: successfully continued agent tapes (of type `FormFillerTape`)
- `user_failed_tapes.yaml`: agent tapes that could not be continued
- `counters.json`: the number of tapes in each file


## Generating synthetic dialogues (tapes)

Next we describe how to run the user & agent in a loop until a fixed number of turns to generate complete synthetic dialogues.

Synthetic data can be used for:
- Evaluating the teacher form-filler agent by simulating its interaction with users at different points in a conversation.
- Distilling a teacher form-filler agent using a large LLM (complex and expensive), into a "student" agent using a smaller LLM (simpler and cheaper).

We generate synthetic dialogues by alternating between calling the Form-filler Agent and User Agent to continue the dialogue tape.

For instance, to generate trees using the [tree_config6](https://github.com/ServiceNow/TapeAgents/blob/formfiller/examples/form_filler/conf/make_tape_tree/tree_config6.yaml) configuration on the `FlyCorp` domain, with the **teacher** agent, 
run the following command (in a tmux window)
```bash
python -m examples.form_filler.dev.make_tape_tree \
  make_tape_tree=tree_config_small \
  make_tape_tree.output_path=.local/make_tape_tree \
  make_tape_tree.preambles_path=examples/form_filler/forms/train/FlyCorp \
  make_tape_tree.global_count=5 \
  agent@make_tape_tree.agent=teacher_agent
```
This will generate a 8-layer "dialogue tree" with a width of 5 (number of tapes per dialogue turn aka per "layer").
User Agents are randomly sampled according to a distribution defined in [tree_config6](https://github.com/ServiceNow/TapeAgents/blob/formfiller/examples/form_filler/conf/make_tape_tree/tree_config6.yaml).
By default this script will use the [openrouter_llama3_405b_temp1](https://github.com/ServiceNow/TapeAgents/blob/formfiller/examples/form_filler/conf/llm/openrouter_llama3_405b_temp1.yaml) llm config which uses openrouter to call llama405B for **both** the teacher and the user agents.
To use this model you must have an openrouter API key saved into your `TAPEAGENTS_LLM_TOKEN` environment variable.
You can check the pricing of this model on the [openrouter website](https://openrouter.ai/meta-llama/llama-3.1-405b-instruct).
You may change the user/agent LLM by overriding `llm@make_tape_tree.agent.llm` or `make_tape_tree.user_simulator_llm` (see Hydra documentation).
Use `force_restart_idx=-1` to automatically resume from the last layer, or a non-negative value to restart from layer `force_restart_idx` (this will discard all subsequent layers, including the current one).

The resulting folder structure in the `output_path` is a follows:
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
formfiller_agent_tapes.yaml          # list of FormFiller tapes that the agent successfully continued
formfiller_agent_tape_failures.yaml  # list of FormFiller tapes that the agent failed to continue
formfiller_user_tapes.yaml           # list of FormFiller tapes that the user successfully continued
user_simulator_tapes.yaml            # list of UserSimulator tapes that the user generated
stats.json                           # counts of last step types for each tape in each of the above files
```

Each layer corresponds to a User or Form-filler Agent turn. Even layers (`layer_0, layer_2, ...`) end with Users turns, while Odd layers (`layer_1, layer_3, ...`) end with Form-Filler agent turns:
- `data.yaml` contains successful user/agent completions, stored as `FormFillerTape` tapes.
- `failures.yaml` contain the dialogues (tapes) that could not be continued, stored as `FormFillerTape`.
- Additionally, for even layers, `user_simulator_tapes` contains tapes that are internal to the User Simulator, stored as `UserSimulatorTape`,

For convenience, we merge all agent (resp. user) dialogues into a single file `formfiller_agent_tapes.yaml` (resp. `formfiller_user_tapes.yaml`)

You can visualize the generated dialogues by running
```bash
python -m examples.form_filler.dev.visualize_formfiller_tapes .local/make_tape_tree/<layer_idx> <optional: give path to tapedata.sqlite to visualize prompts>
```

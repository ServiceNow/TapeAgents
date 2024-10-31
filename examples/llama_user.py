import json
from typing import Literal

from tapeagents.agent import ObservationMaker
from tapeagents.core import AgentEvent, MakeObservation, Observation, Prompt, Tape
from tapeagents.dialog_tape import (
    DialogTape,
    SystemStep,
    UserStep,
)
from tapeagents.llms import LLMStream, TrainableLLM

USER_MODEL_TEMPLATE = """You will generate the next user message in the following conversation.

{conversation}

In the next user message the user will {user_model_prompt}

Output the next message using this template: 

{{
    "role": "user",
    "content": "..."
}}

ONLY output this JSON and nothing else."""


class UserModelFreeFormThought(Observation):
    kind: Literal["user_model_free_form_thought"] = "user_model_free_form_thought"
    content: str


class UserModelInstruction(Observation):
    kind: Literal["user_model_instruction"] = "user_model_instruction"
    instruction: str


UserModelTape = Tape[DialogTape, MakeObservation[UserStep] | UserModelFreeFormThought | UserModelInstruction]
UserModelEvent = AgentEvent[UserModelTape]


class UserModel(ObservationMaker[DialogTape, UserModelTape]):
    instruction: str

    def make_own_tape(self, tape: DialogTape) -> UserModelTape:
        return UserModelTape(context=tape).append(UserModelInstruction(instruction=self.instruction))


class LLAMAUserModel(UserModel):
    def make_prompt(self, tape: UserModelTape):
        (instruction_step,) = tape.steps
        assert isinstance(instruction_step, UserModelInstruction)
        assert tape.context is not None
        conversation = json.dumps(tape.context.model_dump()["steps"], indent=2)
        user_message = USER_MODEL_TEMPLATE.format(
            conversation=conversation, user_model_prompt=instruction_step.instruction
        )
        return Prompt(messages=[{"role": "user", "content": user_message}])

    def generate_steps(self, _, llm_stream: LLMStream):
        yield MakeObservation(new_observation=UserStep.model_validate_json(llm_stream.get_text()))


def try_llama_user_model(llm: TrainableLLM):
    tape = DialogTape(
        context=None,
        steps=[
            SystemStep(content="Respond to the user using the style of Shakespeare books. Be very brief, 5 words max."),
            UserStep(content="Hello, how are you bro?"),
            SystemStep(content="kind stranger, I do fare most well"),
        ],
    )
    llama_user_model1 = LLAMAUserModel(instruction="ask for some dating advice", llms={"default": llm})
    llama_user_model2 = LLAMAUserModel(
        instruction="repeats the USER's previous message and adds 'yay' at the end", llms={"default": llm}
    )

    own_tape1 = llama_user_model1.run(llama_user_model1.make_own_tape(tape)).get_final_tape()
    new_tape = llama_user_model1.add_observation(tape, own_tape1)
    print("--- CHECK GENERATED USER MESSAGE 1 ---")
    print(new_tape.steps[-1])

    print("--- CHECK ORIGINAL TAPE ---")
    print(json.dumps(tape.model_dump(), indent=2))
    print("--- CHECK NEW TAPE")

    print(json.dumps(new_tape.model_dump(), indent=2))

    print("--- CHECK TRACES ---")
    for trace in llama_user_model1.make_training_data(own_tape1):
        print("<CONTEXT>", trace.prompt_text, sep="")
        print("<COMPLETION>", trace.output_text, sep="")

    print("--- CHECK GENERATED USER MESSAGE 2 ---")
    new_tape = llama_user_model2.continue_tape(new_tape)
    print(new_tape.steps[-1])


if __name__ == "__main__":
    try_llama_user_model(
        TrainableLLM(
            base_url="https://api.together.xyz",
            model_name="meta-llama/Llama-3-8b-chat-hf",
            parameters=dict(temperature=0.7, max_tokens=512),
        )
    )

import json
import os
import tempfile

from tapeagents.agent import Agent
from tapeagents.core import PartialStep, Prompt, Tape, TapeMetadata, TrainingText
from tapeagents.dialog_tape import AssistantStep, DialogTape, SystemStep, UserStep
from tapeagents.llms import LLM, LLMOutput, LLMStream, TrainableLLM
from tapeagents.prompting import tape_to_messages


class LLAMAChatBot(Agent[DialogTape]):
    """
    Example of an agent that responds to user messages using the LLAMA model.
    """

    def make_prompt(self, tape: DialogTape):
        return Prompt(messages=tape_to_messages(tape))

    def generate_steps(self, tape: Tape, llm_stream: LLMStream):
        buffer = []
        for event in llm_stream:
            if event.chunk:
                buffer.append(event.chunk)
                yield PartialStep(step=AssistantStep(content="".join(buffer)))
            elif (m := event.output) and isinstance(m, LLMOutput):
                yield AssistantStep(content=m.content or "")
                return
            else:
                raise ValueError(f"Uknown event type from LLM: {event}")
        raise ValueError("LLM didn't return completion")

    def make_llm_output(self, tape: DialogTape, index: int) -> LLMOutput:
        if not isinstance(step := tape.steps[index], AssistantStep):
            raise ValueError(f"Can only make completion for AssistantStep, got {step}")
        return LLMOutput(content=step.content)


def try_llama_chatbot(llm: LLM):
    agent = LLAMAChatBot.create(llm, name="llamachatbot")
    print("--- CHECK CHATTING ---")
    user_messages = [
        "Hello, how are you?",
        "Can you help me design awesome agent framework?",
        "How many parameters do you have?",
    ]

    tape = DialogTape(
        context=None,
        steps=[
            SystemStep(content="Respond to the user using the style of Shakespeare books. Be very brief, 50 words max.")
        ],
    )
    with open("start_tape.json", "w") as f:
        json.dump(tape.model_dump(), f, indent=2)
    tape_db: dict = {tape.metadata.id: tape}
    for message in user_messages:
        tape_with_user_message = tape.append(UserStep(content=message))
        tape_with_user_message.metadata = TapeMetadata(parent_id=tape.metadata.id)
        tape_db[tape_with_user_message.metadata.id] = tape_with_user_message
        print("  User:", message)
        print("  Agent: ", end="")
        n_printed = 0
        for event in agent.run(tape_with_user_message):
            if event.partial_step:
                assert isinstance(step := event.partial_step.step, AssistantStep)
                print(step.content[n_printed:], end="", flush=True)
                n_printed = len(step.content)
            if event.final_tape:
                tape = event.final_tape
                tape_db[tape.metadata.id] = tape
                print()
                print(f"Received new tape of length {len(tape)}")

    print("--- CHECK TRACES ---")
    traces: list[TrainingText] = []
    for i, trace in enumerate(agent.make_training_data(tape)):
        print(f"TRACE {i}")
        print("CONTEXT", trace.prompt_text)
        print("COMPLETION", trace.output_text)
        traces.append(trace)
    with open("traces.json", "w") as f:
        json.dump([t.model_dump() for t in traces], f, indent=2)

    print("--- CHECK SERIALIZATION AND TAPE TREE TRAVERSAL ---")
    cur_tape = tape
    while cur_tape:
        print(json.dumps(cur_tape.model_dump(), indent=2))
        with open("tape.json", "w") as f:
            json.dump(cur_tape.model_dump(), f, indent=2)
        cur_tape = tape_db.get(cur_tape.metadata.parent_id, None)

    print("--- CHECK DESERIALIZATION ---")
    reconstructed_tape = DialogTape.model_validate(tape.model_dump())
    assert reconstructed_tape == tape
    print("didn't crash, we are good")


if __name__ == "__main__":
    tmpdir = tempfile.mkdtemp()
    os.chdir(tmpdir)
    try_llama_chatbot(
        TrainableLLM(
            base_url="https://api.together.xyz",
            model_name="meta-llama/Meta-Llama-3-70B-Instruct-Turbo",
            tokenizer_name="meta-llama/Meta-Llama-3-70B-Instruct",
            parameters=dict(temperature=0.7, max_tokens=512),
            stream=True,
        )
    )
    print("Saved test data to", tmpdir)

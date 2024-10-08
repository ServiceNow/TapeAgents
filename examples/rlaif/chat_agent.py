import json
import os
import tempfile

from tapeagents.agent import Agent
from tapeagents.core import LLMOutput, PartialStep, Prompt, Tape, TapeMetadata, TrainingText
from tapeagents.dialog_tape import AssistantStep, DialogTape, SystemStep, UserStep
from tapeagents.llms import LLM, LLMStream, TrainableLLM
from tapeagents.prompting import tape_to_messages
from tapeagents.environment import Environment
from tapeagents.batch import batch_main_loop

class ChatAgent(Agent[DialogTape]):
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
    
    def make_training_data(self, tape: Tape) -> list[TrainingText]:
        """
        We only train on the last completion
        """
        _, llm_calls = self.reuse(tape)
        return [self.make_training_text(llm_calls[-1], compute_log_probs=True)]


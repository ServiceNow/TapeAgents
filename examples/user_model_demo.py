import json
import logging
from typing import Generator

from langchain_community.tools.tavily_search import TavilySearchResults

from examples.llama_user import UserModel, UserModelEvent, UserModelInstruction, UserModelTape
from tapeagents.core import MakeObservation, Prompt
from tapeagents.demo import Demo
from tapeagents.dialog_tape import (
    DialogContext,
    DialogTape,
    UserStep,
)
from tapeagents.environment import ToolEnvironment
from tapeagents.llms import LiteLLM, LLMOutput
from tapeagents.renderers import render_dialog_plain_text
from tapeagents.renderers.basic import BasicRenderer

from .annotator import GroundednessAnnotator
from .llama_user import USER_MODEL_TEMPLATE
from .openai_function_calling import FunctionCallingAgent

logging.basicConfig(level=logging.INFO)


class DemoUserModel(UserModel):
    def __init__(self, llm: LiteLLM, instruction: str):
        self.llm = llm
        super().__init__(instruction=instruction)

    def render_prompt(self, tape: UserModelTape) -> Prompt:
        (instruction_step,) = tape.steps
        assert isinstance(instruction_step, UserModelInstruction)
        conversation = render_dialog_plain_text(tape.context)
        assitant_message = USER_MODEL_TEMPLATE.format(
            conversation=conversation, user_model_prompt=instruction_step.instruction
        )
        return Prompt(messages=[{"role": "assistant", "content": assitant_message}])

    def generate_events(self, prompt: Prompt) -> Generator[UserModelEvent, None, None]:
        # say we don't need streaming for the agent model here
        for event in self.llm.generate(prompt):
            if m := event.output:
                assert isinstance(m, LLMOutput)
                try:
                    result = json.loads(m.content)
                except Exception:
                    raise ValueError(f"User model LLM returned invalid JSON: {event.output}")
                yield UserModelEvent(step=MakeObservation(new_observation=UserStep(**result)))
                return
        raise ValueError("User model LLM didn't return completion")


def user_model_demo():
    small_llm = LiteLLM(model_name="gpt-3.5-turbo")
    big_llm = LiteLLM(model_name="gpt-4-turbo")
    agent = FunctionCallingAgent(llms={"default": small_llm})
    environment = ToolEnvironment(tools=[TavilySearchResults()])
    init_dialog = DialogTape(context=DialogContext(tools=environment.get_tool_schemas()), steps=[])
    user_models = [
        DemoUserModel(big_llm, "ask about a celebrity"),
        DemoUserModel(big_llm, "ask about a movie"),
    ]
    demo = Demo(
        agent,
        init_dialog,
        environment,
        BasicRenderer(),
        user_models={um.instruction: um for um in user_models},
        annotator=GroundednessAnnotator(llms={"default": big_llm}),
    )
    demo.launch()


if __name__ == "__main__":
    user_model_demo()


from tapeagents.agent import Agent
from tapeagents.core import Prompt
from tapeagents.dialog_tape import AssistantStep, DialogTape, SystemStep, UserStep
from tapeagents.llms import LLMStream, LiteLLM


class MyFirstAgent(Agent[DialogTape]):
    def make_prompt(self, tape: DialogTape) -> Prompt:
        """
        Render tape into the prompt, each step is converted into a message
        """
        return Prompt(messages=tape.as_prompt_messages())

    def generate_steps(self, tape: DialogTape, llm_stream: LLMStream):
        """
        Generate single tape step from the LLM output messages stream.
        """
        yield AssistantStep(content=llm_stream.get_text())

llm = LiteLLM(model_name="gpt-4o-mini-2024-07-18")
agent = MyFirstAgent.create(llm)

# Tape is a sequence of steps that contains all the interactions between the user and the agent happened during the session.
# Let's provide the agent with the description of the task and the first step to start the conversation.
start_tape = DialogTape(
    steps=[
        SystemStep(content="Respond to the user using the style of Shakespeare books. Be very brief, 50 words max."),
        UserStep(content="Hello, tell me about Montreal."),
    ],
)
final_tape = agent.run(start_tape).get_final_tape()
print(f"Final tape: {final_tape.model_dump_json(indent=2)}")
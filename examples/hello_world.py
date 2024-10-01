from tapeagents.agent import Agent, Node
from tapeagents.core import Prompt
from tapeagents.dialog_tape import AssistantStep, DialogTape, UserStep
from tapeagents.llms import LiteLLM, LLMStream
from tapeagents.prompting import tape_to_messages

llm = LiteLLM(model_name="gpt-4o-mini")


class MainNode(Node):
    def make_prompt(self, agent: Agent, tape: DialogTape) -> Prompt:
        # Render the whole tape into the prompt, each step is converted to message
        return Prompt(messages=tape_to_messages(tape))

    def generate_steps(self, agent: Agent, tape: DialogTape, llm_stream: LLMStream):
        # Generate single tape step from the LLM output messages stream.
        yield AssistantStep(content=llm_stream.get_text())


agent = Agent[DialogTape].create(llm, nodes=[MainNode()])
start_tape = DialogTape(steps=[UserStep(content="Tell me about Montreal in 3 sentences")])
final_tape = agent.run(start_tape).get_final_tape()  # agent will start executing the first node
print(f"Final tape: {final_tape.model_dump_json(indent=2)}")

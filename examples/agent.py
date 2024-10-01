import sys

from tapeagents.agent import Agent, Node
from tapeagents.core import Prompt, SetNextNode, Tape
from tapeagents.dialog_tape import AssistantStep, AssistantThought, DialogTape, UserStep
from tapeagents.llms import LLM, LLMStream, TrainableLLM
from tapeagents.prompting import prompt_with_guidance, tape_to_messages


def hello_world(llm: LLM):
    """
    This example demonstrates a simple agent that first thinks and then says "Hello, world!" in the style of Shakespeare.
    """
    agent = Agent.create(
        llm,
        nodes=[
            Node(name="think")
            .with_prompt(
                lambda agent, tape: Prompt(
                    messages=[s.llm_dict() for s in tape.steps]
                    + [{"role": "user", "content": "Describe how Shakespeare would say hello world"}]
                )
            )
            .with_generate_steps(
                lambda agent, tape, llm_stream: (yield AssistantThought(content=llm_stream.get_text()))
            ),
            Node(name="respond")
            .with_prompt(
                lambda agent, tape: Prompt(
                    messages=[s.llm_dict() for s in tape.steps]
                    + [{"role": "user", "content": "Respond with the hello world in the described style"}]
                )
            )
            .with_generate_steps(lambda agent, tape, llm_stream: (yield AssistantStep(content=llm_stream.get_text()))),
        ],
    )
    start_tape = DialogTape(steps=[UserStep(content="Hi!")])
    print(agent.run(start_tape).get_final_tape().model_dump_json(indent=2))


def control_nodes():
    def router(agent: Agent, tape: Tape, llm_stream: LLMStream):
        if tape[-1].content == "Go left":  # type: ignore
            yield SetNextNode(next_node=1)
        elif tape[-1].content == "Go right":  # type: ignore
            yield SetNextNode(next_node=2)
        else:
            yield SetNextNode(next_node=3)

    agent = Agent(
        nodes=[
            Node(name="router").with_generate_steps(router),
            Node(name="go_left").with_fixed_steps([AssistantStep(content="You went left!"), SetNextNode(next_node=0)]),
            Node(name="go_right").with_fixed_steps(
                [AssistantStep(content="You went right!"), SetNextNode(next_node=0)]
            ),
            Node(name="something_else").with_fixed_steps(
                [AssistantStep(content="What do you mean?"), SetNextNode(next_node=0)]
            ),
        ]
    )

    user_messages = ["Go left", "Go right", "Do a kick-flip!"]
    tape = DialogTape(context=None, steps=[])
    for msg in user_messages:
        tape = tape.append(UserStep(content=msg))
        tape = agent.run(tape).get_final_tape()
    print(tape.model_dump_json(indent=2))


def classy_hello_world(llm: LLM):
    class ThinkingNode(Node):
        def make_prompt(self, agent, tape: Tape) -> Prompt:
            return prompt_with_guidance(tape, "Describe how Shakespeare would say hello world")

        def generate_steps(self, agent, tape: Tape, llm_stream: LLMStream):
            yield AssistantThought(content=llm_stream.get_text())

    class RespondingNode(Node):
        def make_prompt(self, agent, tape: Tape) -> Prompt:
            return prompt_with_guidance(tape, "Respond with the hello world in the described style")

        def generate_steps(self, agent, tape: Tape, llm_stream: LLMStream):
            yield AssistantStep(content=llm_stream.get_text())

    agent = Agent.create(llm, nodes=[ThinkingNode(), RespondingNode()])
    start_tape = DialogTape(steps=[UserStep(content="Hi!")])
    print(agent.run(start_tape).get_final_tape().model_dump_json(indent=2))


if __name__ == "__main__":
    llm = TrainableLLM(
        base_url="https://api.together.xyz",
        model_name="meta-llama/Meta-Llama-3-70B-Instruct-Turbo",
        tokenizer_name="meta-llama/Meta-Llama-3-70B-Instruct",
        parameters=dict(temperature=0.7, max_tokens=512),
    )
    match sys.argv[1:]:
        case []:
            hello_world(llm)
        case ["classy"]:
            classy_hello_world(llm)
        case ["control_nodes"]:
            control_nodes()
        case _:
            raise Exception(
                'Usage: TAPEAGENTS_LLM_TOKEN="<your_together_ai_token>" python -m examples.nodes [classy|control_nodes]'
            )

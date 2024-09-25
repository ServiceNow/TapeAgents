import sys

from tapeagents.agent import Agent, Node
from tapeagents.core import Jump, Prompt, Tape
from tapeagents.dialog_tape import AssistantStep, AssistantThought, DialogTape, UserStep
from tapeagents.llms import LLAMA, LLM, LLMStream
from tapeagents.view import TapeView


def hello_world(llm: LLM):
    """
    This example demonstrates a simple agent that first thinks and then says "Hello, world!" in the style of Shakespeare.
    """
    agent = Agent.create(
        llm,
        flow=[
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


def control_flow():
    def router(agent: Agent, tape: Tape, llm_stream: LLMStream):
        if tape[-1].content == "Go left":  # type: ignore
            yield Jump(next_node=1)
        elif tape[-1].content == "Go right":  # type: ignore
            yield Jump(next_node=2)
        else:
            yield Jump(next_node=3)

    agent = Agent(
        flow=[
            Node(name="router").with_generate_steps(router),
            Node(name="go_left").with_fixed_steps([AssistantStep(content="You went left!"), Jump(next_node=0)]),
            Node(name="go_right").with_fixed_steps([AssistantStep(content="You went right!"), Jump(next_node=0)]),
            Node(name="something_else").with_fixed_steps(
                [AssistantStep(content="What do you mean?"), Jump(next_node=0)]
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
        name: str = "think"

        def make_prompt(self, agent, tape: Tape) -> Prompt:
            messages = tape.steps + [UserStep(content="Describe how Shakespeare would say hello world")]
            return Prompt(messages=[m.llm_dict() for m in messages])

        def generate_steps(self, agent, tape: Tape, llm_stream: LLMStream):
            yield AssistantThought(content=llm_stream.get_text())

    class RespondingNode(Node):
        name: str = "respond"

        def make_prompt(self, agent, tape: Tape) -> Prompt:
            messages = tape.steps + [UserStep(content="Respond with the hello world in the described style")]
            return Prompt(messages=[m.llm_dict() for m in messages])

        def generate_steps(self, agent, tape: Tape, llm_stream: LLMStream):
            yield AssistantStep(content=llm_stream.get_text())

    agent = Agent.create(llm, flow=[ThinkingNode(), RespondingNode()])
    start_tape = DialogTape(steps=[UserStep(content="Hi!")])
    print(agent.run(start_tape).get_final_tape().model_dump_json(indent=2))


if __name__ == "__main__":
    llm = LLAMA(
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
        case ["control_flow"]:
            control_flow()
        case _:
            raise Exception(
                'Usage: TAPEAGENTS_LLM_TOKEN="<your_together_ai_token>" python -m examples.flow [classy|control_flow]'
            )

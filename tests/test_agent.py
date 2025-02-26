import copy

import pytest
from pydantic import SerializeAsAny

from tapeagents.agent import DEFAULT, Agent, AgentEvent, AgentStream, Node
from tapeagents.core import Action, AgentStep, PartialStep, Prompt, StepMetadata, Tape, Thought
from tapeagents.llms import LLMStream, MockLLM

MockTape = Tape[None, Action]
MockAgentEvent = AgentEvent[MockTape]


class MockAgent(Agent):
    pass


class MockPrompt(Prompt):
    pass


class EmptyLLM(MockLLM):
    def generate(self, prompt):
        return LLMStream(None, prompt)


def test_get_final_tape_success():
    final_tape = MockTape(steps=[Action()])

    events = iter([MockAgentEvent(), MockAgentEvent(final_tape=final_tape)])
    agent_stream = AgentStream[MockTape](events)  # type: ignore

    assert agent_stream.get_final_tape() == final_tape


def test_get_final_tape_failure():
    events = iter([MockAgentEvent(), MockAgentEvent()])
    agent_stream = AgentStream(events)  # type: ignore

    with pytest.raises(ValueError, match="Agent didn't produce final tape"):
        agent_stream.get_final_tape()


def test_node_make_prompt():
    class MockNode(Node):
        def make_prompt(self, agent, tape):
            return MockPrompt()

    node = MockNode()
    agent = MockAgent()
    tape = MockTape()

    prompt = node.make_prompt(agent, tape)

    assert isinstance(prompt, MockPrompt)


def test_node_generate_steps():
    class MockNode(Node):
        def generate_steps(self, agent, tape, llm_stream):
            yield PartialStep(step=Action())
            yield Action()

    node = MockNode()
    agent = MockAgent()
    tape = MockTape()
    llm_stream = LLMStream(None, Prompt())

    steps = list(node.generate_steps(agent, tape, llm_stream))

    assert len(steps) == 2
    assert isinstance(steps[0], PartialStep)
    assert isinstance(steps[1], Action)


def test_model_post_init_no_name():
    agent = MockAgent()
    agent.model_post_init(None)
    assert agent.name == "MockAgent"


def test_model_post_init_duplicate_subagent_name():
    subagent1 = MockAgent(name="subagent")
    subagent2 = MockAgent(name="subagent")

    with pytest.raises(ValueError, match='Duplicate subagent name "subagent" in subagent 1'):
        MockAgent(subagents=[subagent1, subagent2])


def test_model_post_init_subagent_already_has_manager():
    subagent = MockAgent(name="subagent")
    subagent._manager = MockAgent(name="manager")

    with pytest.raises(ValueError, match="Agent is already a subagent of another agent. Make a copy of your agent."):
        MockAgent(subagents=[subagent])


def test_model_post_init_subagent_not_instance_of_agent():
    subagent = "not_an_agent"

    with pytest.raises(ValueError, match="Subagents must be instances of Agent"):
        MockAgent(subagents=[subagent])


def test_model_post_init_duplicate_node_name():
    node1 = Node(name="node")
    node2 = Node(name="node")

    with pytest.raises(ValueError, match='Duplicate node name "node" in node 1'):
        Agent(nodes=[node1, node2])


def test_create_with_llm():
    llm = MockLLM()
    agent = MockAgent.create(llm)

    assert isinstance(agent, MockAgent)
    assert agent.llms[DEFAULT] == llm


def test_create_with_templates():
    template = "template_string"
    agent = MockAgent.create(templates=template)

    assert isinstance(agent, MockAgent)
    assert agent.templates[DEFAULT] == template


def test_create_with_llms_and_templates():
    llm = MockLLM()
    template = "template_string"
    agent = MockAgent.create(llm, templates=template)

    assert isinstance(agent, MockAgent)
    assert agent.llms[DEFAULT] == llm
    assert agent.templates[DEFAULT] == template


def test_create_with_kwargs():
    agent = MockAgent.create(name="test_agent", max_iterations=10)

    assert isinstance(agent, MockAgent)
    assert agent.name == "test_agent"
    assert agent.max_iterations == 10


def test_create_with_dict_llms_and_templates():
    llms = {"default": MockLLM(), "llm2": MockLLM()}
    templates = {"template1": "template_string1", "template2": "template_string2"}
    agent = MockAgent.create(llms, templates=templates)  # type: ignore

    assert isinstance(agent, MockAgent)
    assert agent.llms == llms
    assert agent.templates == templates


def test_make_prompt():
    class NodeMockPrompt(Prompt):
        pass

    class MockNode(Node):
        def make_prompt(self, agent, tape):
            return NodeMockPrompt()

    class MockAgent(Agent):
        def select_node(self, tape):
            return MockNode()

    agent = MockAgent()
    tape = MockTape()

    prompt = agent.make_prompt(tape)

    assert isinstance(prompt, NodeMockPrompt)


def test_generate_steps():
    class MockNode(Node):
        def generate_steps(self, agent, tape, llm_stream):
            yield PartialStep(step=Action())
            yield Action()

    class MockAgent(Agent):
        def select_node(self, tape):
            return MockNode()

    agent = MockAgent()
    tape = MockTape()
    llm_stream = LLMStream(None, Prompt())

    steps = list(agent.generate_steps(tape, llm_stream))

    assert len(steps) == 2
    assert isinstance(steps[0], PartialStep)
    assert isinstance(steps[1], Action)


def test_run_iteration_with_llm_stream():
    class MockNode(Node):
        def generate_steps(self, agent, tape, llm_stream):
            yield PartialStep(step=Action())
            yield Action()

    class MockAgent(Agent):
        def select_node(self, tape):
            return MockNode()

    agent = MockAgent()
    tape = MockTape()
    llm_stream = LLMStream(None, Prompt())

    steps = list(agent.run_iteration(tape, llm_stream))

    assert len(steps) == 2
    assert isinstance(steps[0], PartialStep)
    assert isinstance(steps[1], Action)


def test_run_iteration():
    class MockNode(Node):
        def generate_steps(self, agent, tape, llm_stream):
            yield PartialStep(step=Action())
            yield Action()

    class MockAgent(Agent):
        def select_node(self, tape):
            return MockNode()

    agent = MockAgent(llms={DEFAULT: EmptyLLM()})
    tape = MockTape()

    steps = list(agent.run_iteration(tape))

    assert len(steps) == 2
    assert isinstance(steps[0], PartialStep)
    assert isinstance(steps[1], Action)


def test_run_iteration_with_multiple_llms():
    class MockNode(Node):
        def generate_steps(self, agent, tape, llm_stream):
            yield PartialStep(step=Action())
            yield Action()

    class MockAgent(Agent):
        def select_node(self, tape):
            return MockNode()

    agent = MockAgent(llms={"default": EmptyLLM(), "llm2": EmptyLLM()})
    tape = MockTape()

    for step in agent.run_iteration(tape):
        assert isinstance(step, (PartialStep, Action))
        if not isinstance(step, PartialStep):
            assert step.metadata.llm == "default"


def test_run_iteration_with_agent_step():
    class MockNode(Node):
        def generate_steps(self, agent, tape, llm_stream):
            yield AgentStep(metadata=StepMetadata())

    class MockAgent(Agent):
        nodes: list[SerializeAsAny[Node]] = [MockNode(name="node1")]

        def select_node(self, tape):
            return self.nodes[0]

        def get_node_name(self, tape) -> str:
            return self.nodes[0].name

    agent = MockAgent(llms={DEFAULT: EmptyLLM()})
    tape = MockTape()

    steps = list(agent.run_iteration(tape))

    assert len(steps) == 1
    assert isinstance(steps[0], AgentStep)
    assert steps[0].metadata.prompt_id is not None
    assert steps[0].metadata.node == "node1"


def test_select_node_explicit_next_node():
    class MockView:
        next_node = "node2"
        last_node = None

    class MockTapeViewStack:
        top = MockView()

    class MockAgent(Agent):
        def compute_view(self, tape):
            return MockTapeViewStack()

        def find_node(self, name):
            return Node(name=name)

    agent = MockAgent(nodes=[Node(name="node1"), Node(name="node2")])
    tape = MockTape()

    selected_node = agent.select_node(tape)

    assert selected_node.name == "node2"


def test_select_node_no_nodes_run_yet():
    class MockView:
        next_node = None
        last_node = None

    class MockTapeViewStack:
        top = MockView()

    class MockAgent(Agent):
        def compute_view(self, tape):
            return MockTapeViewStack()

    agent = MockAgent(nodes=[Node(name="node1"), Node(name="node2")])
    tape = MockTape()

    selected_node = agent.select_node(tape)

    assert selected_node.name == "node1"


def test_select_node_select_next_node():
    class MockView:
        next_node = None
        last_node = "node1"

    class MockTapeViewStack:
        top = MockView()

    class MockAgent(Agent):
        def compute_view(self, tape):
            return MockTapeViewStack()

    agent = MockAgent(nodes=[Node(name="node1"), Node(name="node2")])
    tape = MockTape()

    selected_node = agent.select_node(tape)

    assert selected_node.name == "node2"


def test_select_node_next_node_not_found():
    class MockView:
        next_node = None
        last_node = "node2"

    class MockTapeViewStack:
        top = MockView()

    class MockAgent(Agent):
        def compute_view(self, tape):
            return MockTapeViewStack()

    agent = MockAgent(nodes=[Node(name="node1"), Node(name="node2")])
    tape = MockTape()

    with pytest.raises(ValueError, match="Next node not found"):
        agent.select_node(tape)


def test_run_metadata():
    class MockNode(Node):
        def generate_steps(self, agent, tape, llm_stream):
            yield Thought()
            yield Action()

    class MockAgent(Agent):
        def select_node(self, tape):
            return MockNode()

    agent = MockAgent(llms={DEFAULT: EmptyLLM()})
    tape = MockTape()

    initial_tape_metadata = copy.deepcopy(tape.metadata)
    assert initial_tape_metadata.n_added_steps == 0

    final_tape = agent.run(tape).get_final_tape()

    # check that the original tape metadata is the same
    assert tape.metadata == initial_tape_metadata
    # check that the new tape metadata is updated correclty
    assert final_tape.metadata.id != initial_tape_metadata.id
    assert final_tape.metadata.parent_id == initial_tape_metadata.id
    assert final_tape.metadata.n_added_steps == 2
    assert final_tape.metadata.author == agent.name
    # assert that the rest is the same, except for id, parent_id, n_added_steps, and author
    initial_tape_metadata.id, final_tape.metadata.id = None, None
    initial_tape_metadata.parent_id, final_tape.metadata.parent_id = None, None
    initial_tape_metadata.n_added_steps, final_tape.metadata.n_added_steps = 0, 0
    initial_tape_metadata.author, final_tape.metadata.author = None, None
    assert initial_tape_metadata == final_tape.metadata

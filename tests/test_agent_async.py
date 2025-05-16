import asyncio
import time
from unittest.mock import MagicMock

import aiohttp
import pytest

from tapeagents.agent import Agent
from tapeagents.core import Prompt
from tapeagents.dialog_tape import DialogTape, UserStep
from tapeagents.llms import LLMEvent, LLMOutput, LLMStream, TrainableLLM
from tapeagents.nodes import StandardNode
from tapeagents.steps import ReasoningThought


class MockAsyncLLM(TrainableLLM):
    """Mock Async LLM interface with 1 second delay in responses"""

    async def agenerate(self, prompt: Prompt, session: aiohttp.ClientSession, **kwargs) -> LLMStream:
        """Simulate LLM response with 1 second delay"""
        await asyncio.sleep(1)  # Simulate 1 second delay

        # Return different responses based on the question
        if "capital of France" in prompt.messages[0]["content"]:
            content = "The capital of France is Paris."
        elif "capital of Germany" in prompt.messages[0]["content"]:
            content = "The capital of Germany is Berlin."
        else:
            content = f"Response to: {prompt}"
        output = LLMOutput(content=content)
        llm_call = self.log_output(prompt, output)

        def _gen():
            yield LLMEvent(llm_call=llm_call, output=output)

        return LLMStream(generator=_gen(), prompt=prompt)

    def count_tokens(self, messages: list[dict] | str) -> int:
        return 1


@pytest.mark.asyncio
async def test_agent_arun():
    """Test the arun method of Agent class"""
    # Create agent with mocked LLM
    llm = MockAsyncLLM(model_name="mock-model")
    agent = Agent.create(
        llms=llm,
        nodes=[
            StandardNode(guidance="lets think step by step"),
            StandardNode(name="verify", guidance="lets verify the answer"),
        ],
    )

    # Create test question and tape
    question = "What is the capital of France?"
    tape = DialogTape(steps=[UserStep(content=question)])

    # Test arun method
    events = []
    mock_session = MagicMock()
    async for event in agent.arun(tape=tape, session=mock_session, max_iterations=2):
        events.append(event)

    # Verify results
    assert len(events) > 2
    step_events = [e for e in events if e.step]
    assert len(step_events) == 2

    # Check both nodes were used
    node_names = [e.step.metadata.node for e in step_events]
    assert "StandardNode" in node_names
    assert "verify" in node_names

    # Check response content
    assert isinstance(step_events[0].step, ReasoningThought)
    assert isinstance(step_events[1].step, ReasoningThought)
    assert "Paris" in step_events[0].step.reasoning
    assert "Paris" in step_events[1].step.reasoning


@pytest.mark.asyncio
async def test_agent_arun_iteration():
    """Test the arun_iteration method of Agent class"""
    # Create agent with mocked LLM
    llm = MockAsyncLLM(model_name="mock-model")
    agent = Agent.create(
        llms=llm,
        nodes=[
            StandardNode(guidance="lets think step by step"),
            StandardNode(name="verify", guidance="lets verify the answer"),
        ],
    )

    # Create test question and tape
    question = "What is the capital of France?"
    tape = DialogTape(steps=[UserStep(content=question)])

    # Test single iteration
    mock_session = MagicMock()
    steps = []
    async for step in agent.arun_iteration(tape=tape, session=mock_session):
        steps.append(step)

    assert len(steps) == 1

    # Verify results
    step = steps[0]
    assert step is not None
    assert step.metadata.node == "StandardNode"
    assert isinstance(step, ReasoningThought)
    assert "Paris" in step.reasoning


@pytest.mark.asyncio
async def test_concurrent_execution():
    """Test concurrent execution of multiple questions"""
    # Create agent with mocked LLM
    llm = MockAsyncLLM(model_name="mock-model")
    agent = Agent.create(
        llms=llm,
        nodes=[
            StandardNode(guidance="lets think step by step"),
            StandardNode(name="verify", guidance="lets verify the answer"),
        ],
    )

    # Set up multiple questions
    questions = [
        "What is the capital of France?",
        "What is the capital of Germany?",
        "How are you?",
        "What is the capital of Italy?" "What is the capital of Spain?",
        "Who is the president of the USA?",
        "How is the weather today?",
        "Can you tell me a joke?",
        "What is the meaning of life?",
        "Hey, what is your name?",
        "What is the capital of Japan?",
    ]
    mock_session = MagicMock()

    # Function to process one question
    async def process_question(question):
        steps = []
        async for event in agent.arun(
            tape=DialogTape(steps=[UserStep(content=question)]),
            session=mock_session,
            max_iterations=2,
        ):
            if event.step:
                steps.append(event.step)
        return question, steps

    # Measure time for concurrent execution
    start_time = time.time()
    results = await asyncio.gather(*[process_question(q) for q in questions])
    elapsed_time = time.time() - start_time

    # Verify results
    assert len(results) == 10

    # Check if both questions were processed correctly
    for question, steps in results:
        assert len(steps) == 2
        if "France" in question:
            assert "Paris" in steps[0].reasoning
            assert "Paris" in steps[1].reasoning
        elif "Germany" in question:
            assert "Berlin" in steps[0].reasoning
            assert "Berlin" in steps[1].reasoning

    assert elapsed_time < 3  # 2 llm requests per question each taking 1 second, plus some overhead
    print(f"Elapsed time {elapsed_time:.2f}")


if __name__ == "__main__":
    asyncio.run(test_concurrent_execution())

import json
from typing import Literal

from tapeagents.core import Action, Tape, TapeMetadata


class DummyStep(Action):
    pass


TestTape = Tape[None, DummyStep]


def test_add_two_tapes():
    tape1 = TestTape(steps=[DummyStep(), DummyStep()])
    tape2 = TestTape(steps=[DummyStep()])

    result = tape1 + tape2

    assert len(result.steps) == 3
    assert isinstance(result, Tape)
    assert result.metadata.n_added_steps == 1


def test_add_tape_with_iterable():
    tape = TestTape(steps=[DummyStep()])
    steps = [DummyStep(), DummyStep()]

    result = tape + steps

    assert len(result.steps) == 3
    assert isinstance(result, Tape)
    assert result.metadata.n_added_steps == 2


def test_add_empty_tape():
    tape1 = TestTape(steps=[DummyStep()])
    tape2 = TestTape(steps=[])

    result = tape1 + tape2

    assert len(result.steps) == 1
    assert isinstance(result, Tape)
    assert result.metadata.n_added_steps == 0


def test_add_to_empty_tape():
    tape = TestTape(steps=[])
    steps = [DummyStep(), DummyStep()]

    result = tape + steps

    assert len(result.steps) == 2
    assert isinstance(result, Tape)
    assert result.metadata.n_added_steps == 2


def test_getitem_single_step():
    tape = TestTape(steps=[DummyStep(), DummyStep(), DummyStep()])

    step = tape[1]

    assert isinstance(step, DummyStep)
    assert step == tape.steps[1]


def test_getitem_slice():
    tape = TestTape(steps=[DummyStep(), DummyStep(), DummyStep()], metadata=TapeMetadata(author="test"))

    sliced_tape = tape[1:3]

    assert isinstance(sliced_tape, Tape)
    assert len(sliced_tape.steps) == 2
    assert sliced_tape.steps[0] == tape.steps[1]
    assert sliced_tape.steps[1] == tape.steps[2]
    assert sliced_tape.metadata.author is None


def test_getitem_slice_empty():
    tape = TestTape(steps=[DummyStep(), DummyStep(), DummyStep()])

    sliced_tape = tape[3:3]

    assert isinstance(sliced_tape, Tape)
    assert len(sliced_tape.steps) == 0
    assert sliced_tape.metadata.author is None


def test_append_single_step():
    tape = TestTape(steps=[DummyStep()])
    new_step = DummyStep()

    result = tape.append(new_step)

    assert len(result.steps) == 2
    assert result.steps[-1] == new_step
    assert isinstance(result, Tape)
    # check that metadata is the same as a brand new one
    assert result.metadata == TapeMetadata(id=result.metadata.id)


def test_append_multiple_steps():
    tape = TestTape(steps=[DummyStep()])
    new_step1 = DummyStep()
    new_step2 = DummyStep()

    result = tape.append(new_step1).append(new_step2)

    assert len(result.steps) == 3
    assert result.steps[-2] == new_step1
    assert result.steps[-1] == new_step2
    assert isinstance(result, Tape)
    # check that metadata is the same as a brand new one
    assert result.metadata == TapeMetadata(id=result.metadata.id)


def test_append_to_empty_tape():
    tape = TestTape(steps=[])
    new_step = DummyStep()

    result = tape.append(new_step)

    assert len(result.steps) == 1
    assert result.steps[0] == new_step
    assert isinstance(result, Tape)
    # check that metadata is the same as a brand new one
    assert result.metadata == TapeMetadata(id=result.metadata.id)


def test_with_new_id():
    tape = TestTape(steps=[DummyStep(), DummyStep()], metadata=TapeMetadata(author="test"))

    new_tape = tape.with_new_id()

    assert isinstance(new_tape, Tape)
    assert new_tape.metadata.id != tape.metadata.id
    assert new_tape.metadata.author is None
    assert len(new_tape.steps) == len(tape.steps)
    assert new_tape.steps == tape.steps


def test_llm_dict():
    class TestStep(Action):
        kind: Literal["test_step"] = "test_step"
        data: str = "test_data"

    step = TestStep()
    llm_dict = step.llm_dict()

    assert "metadata" not in llm_dict
    assert llm_dict["kind"] == "test_step"
    assert llm_dict["data"] == "test_data"


def test_llm_dict_excludes_none_values():
    class TestStep(Action):
        kind: Literal["test_step"] = "test_step"
        data: str | None = None

    step = TestStep()
    llm_dict = step.llm_dict()

    assert "data" not in llm_dict
    assert llm_dict["kind"] == "test_step"


def test_llm_view():
    class TestStep(Action):
        kind: Literal["test_step"] = "test_step"
        data: str = "test_data"

    step = TestStep()
    llm_view = step.llm_view()

    expected_output = json.dumps(step.llm_dict(), indent=2, ensure_ascii=False)
    assert llm_view == expected_output


def test_llm_view_no_indent():
    class TestStep(Action):
        kind: Literal["test_step"] = "test_step"
        data: str = "test_data"

    step = TestStep()
    llm_view = step.llm_view(indent=None)

    expected_output = json.dumps(step.llm_dict(), indent=None, ensure_ascii=False)
    assert llm_view == expected_output


def test_llm_view_excludes_none_values():
    class TestStep(Action):
        kind: Literal["test_step"] = "test_step"
        data: str | None = None

    step = TestStep()
    llm_view = step.llm_view()

    expected_output = json.dumps(step.llm_dict(), indent=2, ensure_ascii=False)
    assert llm_view == expected_output

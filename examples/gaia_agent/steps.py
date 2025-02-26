import os
import shutil
from typing import Any, Literal, TypeAlias, Union

from pydantic import Field

from tapeagents.core import (
    LLMOutputParsingFailureAction,
    Observation,
    SetNextNode,
    StopStep,
    Tape,
    TapeMetadata,
    Thought,
)
from tapeagents.dialog_tape import DialogContext, UserStep
from tapeagents.environment import CodeExecutionResult, ExecuteCode
from tapeagents.steps import (
    ActionExecutionFailure,
    ImageObservation,
    ReasoningThought,
    VideoObservation,
    WatchVideoAction,
)
from tapeagents.tools.browser import (
    ClickAction,
    GoBackAction,
    GoForwardAction,
    HoverAction,
    InputTextAction,
    MouseClickAction,
    MouseHoverAction,
    OpenUrlAction,
    PageScreenshotObservation,
    PressAction,
    SelectOptionAction,
    TypeTextAction,
)
from tapeagents.tools.calculator import CalculationResultObservation
from tapeagents.tools.code_executor import PythonCodeAction
from tapeagents.tools.simple_browser import PageDownAction, PageObservation, PageUpAction, ReadDocumentAction
from tapeagents.tools.web_search import SearchAction, SearchResultsObservation


class Plan(Thought):
    """
    Thought that contains the plan to follow when answering the question
    """

    kind: Literal["plan_thought"] = "plan_thought"
    plan: list[str] = Field(description="List of steps to follow when answering the question")


class FactsSurvey(Thought):
    """
    Thought that contains the list of facts needed to answer the question
    """

    kind: Literal["facts_survey_thought"] = "facts_survey_thought"
    given_facts: list[str] = Field(
        description="List of facts already provided in the question",
        default=[],
    )
    facts_to_lookup: list[str] = Field(
        description="List of facts that need to be found on the web or in documents",
        default=[],
    )
    facts_to_derive: list[str] = Field(
        description="List of facts that need to be derived from given facts using reasoning or code execution",
        default=[],
    )
    facts_to_guess: list[str] = Field(
        description="List of facts that need to be guessed from given facts, documents, and reasoning",
        default=[],
    )


class ExtractedFacts(Thought):
    """
    Thought that contains the list of facts extracted from the document
    """

    kind: Literal["extracted_facts_thought"] = "extracted_facts_thought"
    extracted_facts: list[str] | dict[str, Any] | str = Field(description="facts extracted from the observation")


class GaiaQuestion(Observation):
    kind: Literal["question"] = "question"
    content: str
    filename: str | None = None

    @classmethod
    def from_task(cls, question: dict):
        question_prompt = question["Question"]
        filename = None
        if question["file_name"]:
            basename = os.path.basename(question["file_name"])
            tmp_fname = f"/tmp/{basename}"
            shutil.copyfile(question["file_name"], tmp_fname)
            assert os.path.exists(tmp_fname)
            filename = tmp_fname
        return cls(content=question_prompt, filename=filename)


class GaiaAnswer(StopStep):
    """
    Action that indicates the agent has finished the plan and contains the answer or description of failure.
    The answer should use already determined facts without additional conversion!
    Your final answer should be a number OR as few words as possible OR a comma-separated list of numbers and/or strings.
    ADDITIONALLY, your final answer MUST follow any formatting instructions specified in the original question (e.g., alphabetization, sequencing, units, rounding, decimal places, etc.)
    If asked for a number, express it numerically, don't use commas, do not add anything after the number, don't include units such as $ or percent signs unless specified otherwise in the question.
    If asked for a string, don't use articles or abbreviations (e.g. for cities), unless specified otherwise. Don't output any final sentence punctuation such as '.', '!', or '?'.
    If asked for a comma-separated list, apply the above rules depending on whether the elements are numbers or strings.
    If unable to determine the final answer, output an empty string.
    """

    kind: Literal["gaia_answer_action"] = "gaia_answer_action"
    success: bool = Field(description="True if the task was successful, False otherwise")
    overview: str = Field(
        description="List of steps performed to answer the question. If the task was not successful, includes the reason for failure"
    )
    answer_unit: str = Field(description="Unit of measurement for the answer, if applicable; otherwise an empty string")
    answer: Any = Field(description="Short final answer")
    long_answer: str = Field(description="Detailed final answer not restricted by format rules")


THOUGHTS = (ReasoningThought, GaiaAnswer)
GaiaStep: TypeAlias = Union[
    ExtractedFacts,
    ClickAction,
    OpenUrlAction,
    GoBackAction,
    GoForwardAction,
    HoverAction,
    InputTextAction,
    MouseClickAction,
    MouseHoverAction,
    TypeTextAction,
    PressAction,
    PageDownAction,
    PageUpAction,
    SelectOptionAction,
    SearchAction,
    PageDownAction,
    ReadDocumentAction,
    PythonCodeAction,
    ReasoningThought,
    SearchAction,
    WatchVideoAction,
    GaiaAnswer,
    GaiaQuestion,
    SearchResultsObservation,
    PageObservation,
    PageScreenshotObservation,
    ImageObservation,
    VideoObservation,
    CalculationResultObservation,
    CodeExecutionResult,
    ActionExecutionFailure,
    Plan,
    FactsSurvey,
    ExecuteCode,
    LLMOutputParsingFailureAction,
    SetNextNode,
    UserStep,
]


class GaiaMetadata(TapeMetadata):
    task: dict = Field(default_factory=dict)
    result: Any = None
    terminated: bool = False
    attempt_number: int = 0
    level: int = 0
    other: dict = Field(default_factory=dict)


class GaiaTape(Tape[DialogContext, GaiaStep]):
    metadata: GaiaMetadata = Field(default_factory=GaiaMetadata)
    context: DialogContext | None = DialogContext(tools=[])

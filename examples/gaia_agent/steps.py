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
from tapeagents.mcp import MCPToolCall, MCPToolResult
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
    Action for task completion, with answer or failure description. Numbers should be plain numeric values without commas or units unless specified.
    Strings should avoid articles/abbreviations unless specified. Lists should be comma-separated.
    Final answer must follow any formatting instructions (units, rounding, decimal places, etc.) from original question.
    Use empty string if unable to determine answer.
    """

    kind: Literal["gaia_answer_action"] = "gaia_answer_action"
    success: bool
    overview: str = Field(description="Short summary of the steps taken, or failure description if unsuccessful")
    answer_unit: str = Field(description="Unit of measurement for the answer or empty string")
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
    MCPToolCall,
    MCPToolResult,
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

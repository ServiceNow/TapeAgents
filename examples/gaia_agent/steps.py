import os
import shutil
from typing import Any, Literal, TypeAlias, Union

from pydantic import Field

from tapeagents.core import LLMOutputParsingFailureAction, Observation, SetNextNode, StopStep, Thought
from tapeagents.environment import CodeExecutionResult, ExecuteCode
from tapeagents.steps import (
    ActionExecutionFailure,
    ImageObservation,
    ReasoningThought,
    VideoObservation,
    WatchVideoAction,
)
from tapeagents.tools.calculator import CalculationResultObservation
from tapeagents.tools.code_executor import PythonCodeAction
from tapeagents.tools.search import SearchAction, SearchResultsObservation
from tapeagents.tools.simple_browser import NextPageAction, PageObservation, ReadDocumentAction


class PlanThought(Thought):
    """
    Thought that contains the plan to follow to answer the question
    """

    kind: Literal["plan_thought"] = "plan_thought"
    plan: list[str] = Field(description="list of steps to follow to answer the question")


class ListOfFactsThought(Thought):
    """
    Thought that contains the list of facts that are needed to answer the question
    """

    kind: Literal["list_of_facts_thought"] = "list_of_facts_thought"
    given_facts: list[str] = Field(
        description="list of facts that are already given in the question",
        default=[],
    )
    facts_to_lookup: list[str] = Field(
        description="list of facts that need to be looked up on the web or in documents",
        default=[],
    )
    facts_to_derive: list[str] = Field(
        description="list of facts that need to be derived from the given facts using reasoning or code execution",
        default=[],
    )
    facts_to_guess: list[str] = Field(
        description="list of facts that need to be guessed from the given facts, documents and reasoning",
        default=[],
    )


class ReadingResultThought(Thought):
    """
    Thought that outputs the result of the reading the document page from the previous step
    """

    kind: Literal["reading_result_thought"] = "reading_result_thought"
    fact_description: str = Field(description="description of the fact that we're looking for in the document")
    fact_found: bool = Field(description="True if the fact was found in the document, False otherwise")
    quote_with_fact: str = Field(
        description="quote from the document that contains the fact, if found, otherwise empty string"
    )
    where_to_look_next: str = Field(
        description="description of where to look next in the document if the fact was not found and there is some hint in the page",
        default="",
    )


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
    Action that indicates that the agent has finished the plan and contains answer or the decsription of failure.
    The answer should use already determined facts without any additional conversion!
    Your final answer should be a number OR as few words as possible OR a comma separated list of numbers and/or strings.
    ADDITIONALLY, your final answer MUST adhere to any formatting instructions specified in the original question (e.g., alphabetization, sequencing, units, rounding, decimal places, etc.)
    If you are asked for a number, express it numerically, don't use commas, do not add anything after the number, don't include units such as $ or percent signs unless specified in question otherwise.
    If you are asked for a string, don't use articles or abbreviations (e.g. for cities), unless specified otherwise. Don't output any final sentence punctuation such as '.', '!', or '?'.
    If you are asked for a comma separated list, apply the above rules depending on whether the elements are numbers or strings.
    If you are unable to determine the final answer, output empty string
    """

    kind: Literal["gaia_answer_action"] = "gaia_answer_action"
    success: bool = Field(description="True if the task was successful, False otherwise")
    overview: str = Field(description="overview of the task")
    answer_unit: str = Field(description="unit of the answer, if applicable, otherwise empty string")
    answer: Any = Field(description="short final answer")


PLAN_STEPS = (PlanThought, ListOfFactsThought)
STEPS_WITHOUT_CODE = (
    ReadingResultThought,
    ReasoningThought,
    SearchAction,
    ReadDocumentAction,
    NextPageAction,
    WatchVideoAction,
    GaiaAnswer,
)
AGENT_STEPS = STEPS_WITHOUT_CODE + (PythonCodeAction,)
OBSERVATIONS = (
    GaiaQuestion,
    SearchResultsObservation,
    PageObservation,
    ImageObservation,
    VideoObservation,
    CalculationResultObservation,
    CodeExecutionResult,
    ActionExecutionFailure,
)
SPECIAL_STEPS = (ExecuteCode, LLMOutputParsingFailureAction, SetNextNode)
GaiaStep: TypeAlias = Union[PLAN_STEPS + AGENT_STEPS + OBSERVATIONS + SPECIAL_STEPS]  # type: ignore

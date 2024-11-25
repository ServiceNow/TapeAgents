import os
import shutil
from typing import Annotated, Any, Literal, TypeAlias, Union

from pydantic import BaseModel, Field

from tapeagents.core import Action, Error, LLMOutputParsingFailureAction, Observation, SetNextNode, StopStep, Thought
from tapeagents.dialog_tape import ImageObservation
from tapeagents.utils import get_step_schemas_from_union_type


################### Base Step Classes ###################
class GaiaThought(Thought):
    pass


class GaiaAction(Action):
    pass


class GaiaObservation(Observation):
    pass


class FactSchema(BaseModel):
    name: str = Field(
        description="fact name, should be unique, lowercase, snake_case, without spaces and special characters"
    )
    description: str = Field(description="short free-form description of the fact")
    format: str = Field(description="format of the fact value, could be float, int, bool, string, list, dict, etc.")
    unit: str = Field(description="unit of the fact value, if applicable, otherwise empty string", default="")


class FactSchemaWithValue(FactSchema):
    value: Any


################### Planning Phase Thoughts ###################


class DraftPlansThought(GaiaThought):
    """
    Thought that contains 3 draft plans to follow to answer the question
    """

    kind: Literal["draft_plans_thought"] = "draft_plans_thought"
    plan: list[str] = Field(description="main list of steps to follow to answer the question")
    alternative_plan: list[str] = Field(description="alternative list of steps to follow to answer the question")
    alternative_plan2: list[str] = Field(
        description="another alternative list of steps to follow to answer the question"
    )


class PlanThought(GaiaThought):
    """
    Thought that contains the plan to follow to answer the question
    """

    kind: Literal["plan_thought"] = "plan_thought"
    plan: list[str] = Field(description="list of steps to follow to answer the question")


class SourcesThought(GaiaThought):
    """
    Thought that contains the sources to use to answer the question. It could be web search, wikipedia, local document path and so on
    """

    kind: Literal["sources_thought"] = "sources_thought"
    sources: dict[str, str] = Field(
        description="dictionary of sources to use to answer the question. Key is the source name, value is the string describing the source"
    )


class ListOfFactsThought(GaiaThought):
    """
    Thought that contains the list of facts that are needed to answer the question
    """

    kind: Literal["list_of_facts_thought"] = "list_of_facts_thought"
    given_facts: list[FactSchemaWithValue] = Field(
        description="list of facts that are already given in the question",
        default=[],
    )
    facts_to_lookup: list[FactSchema] = Field(
        description="list of facts that need to be looked up on the web or in documents",
        default=[],
    )
    facts_to_derive: list[FactSchema] = Field(
        description="list of facts that need to be derived from the given facts using reasoning or code execution",
        default=[],
    )
    facts_to_guess: list[FactSchema] = Field(
        description="list of facts that need to be guessed from the given facts, documents and reasoning",
        default=[],
    )


################### Thoughts ###################


class ReasoningThought(GaiaThought):
    """
    Chain of thoughts of logical reasoning to find the answer. Deductive reasoning could be used to produce a new fact. You can use the facts from the previous steps in the reasoning
    """

    kind: Literal["reasoning_thought"] = "reasoning_thought"
    reasoning: list[str] = Field(description="reasoning sentences that describe how to move forward with the task")


class StartSubtask(GaiaThought):
    """
    Thought that indicates that you start working on the subtask from the plan. You should always finish previous subtask using finish_subtask_thought before starting a new one
    """

    kind: Literal["start_subtask_thought"] = "start_subtask_thought"
    plan_step: str = Field(description="plan step description")


class FinishSubtask(GaiaThought):
    """
    Thought that indicates that you've finished working on the subtask from the plan. You cannot produce that step right after the start subtask, there MUST be some steps in between
    """

    kind: Literal["finish_subtask_thought"] = "finish_subtask_thought"
    plan_step: str = Field(description="plan step description")
    success: bool = Field(description="True if the subtask was successful, False otherwise")
    overview: str = Field(
        description="overview of the subtask. If the subtask was successful, describe the result. If the subtask was unsuccessful, describe the reason for failure"
    )


class NewFactThought(GaiaThought):
    """
    Thought that outputs new fact value, extracted from the previous steps. Value must follow the format described in list_of_facts_thought. If the fact has units that need to be converted, use convert_fact_action after this thought.
    """

    kind: Literal["new_fact_thought"] = "new_fact_thought"
    fact_name: str = Field(
        description="fact name, should be unique, lowercase, snake_case, without spaces and special characters"
    )
    unit: str = Field(description="unit of the fact value, if applicable, otherwise empty string")
    value: Any = Field(
        description="json-parsable value of the fact without units, using format from the list_of_facts_thought"
    )


class ReadingResultThought(GaiaThought):
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


################### Actions ###################
class SearchAction(GaiaAction):
    """
    Action that provides parameters for a search function call. Could search in the web_search or wikipedia. Search results will be ordered by relevance from top to bottom
    """

    kind: Literal["search_action"] = "search_action"
    source: str = Field(description="source to search in, could be web_search or wikipedia")
    query: str = Field(description="search query")


class NextPageAction(GaiaAction):
    """
    Action that returns the next page of the last document
    """

    kind: Literal["next_page_action"] = "next_page_action"


class ReadDocumentAction(GaiaAction):
    """
    Action that loads the document, file, image, video or page from the provided url or file path and returns the first page of its content. To read the following pages use next_page_action
    """

    kind: Literal["read_document_action"] = "read_document_action"
    url: str = Field(description="url of the document")
    fact_description: str = Field(description="description of the fact to look for in the document")
    fact_name: str = Field(description="fact name to look for in the document")


class ConvertFactAction(GaiaAction):
    """
    Action to convert the fact value to the requested unit using math expression. If the unit is already correct, just return the value. Be especially careful when dealing with time facts!
    """

    kind: Literal["convert_fact_action"] = "convert_fact_action"
    original_fact_name: str = Field(description="original fact name")
    converted_fact_name: str = Field(description="fact name from list_of_facts_thought")
    fact_description: str = Field(description="short description of the fact")
    fact_value: Any = Field(description="original value of the fact")
    unit: str = Field(description="original unit of the fact value, if applicable, otherwise empty string")
    requested_unit: str = Field(description="required unit of the fact value")
    reasoning: str = Field(description="explanation of what the conversion expression should do")
    expression: str = Field(description="math expression to convert the value")


class UseCalculatorAction(GaiaAction):
    """
    Action to use calculator to find the new fact. This python math expression uses only the fact names from the previous steps and constants. The expression should be a single line. You can use exp, cos, sin, tan, abs, trunc, sgn, round
    """

    kind: Literal["use_calculator_action"] = "use_calculator_action"
    expression: str = Field(description="math expression using previously known fact names and constants")
    fact_name: str = Field(
        description="fact name to save calculations result, should be unique, lowercase, snake_case, without spaces and special characters"
    )
    fact_unit: str = Field(description="expected unit of the fact value, if applicable, otherwise empty string")
    facts: dict | None = None


class PythonCodeAction(GaiaAction):
    """
    Action to execute the python code snippet.
    """

    kind: Literal["python_code_action"] = "python_code_action"
    code: str = Field(description="snippet of python code with escaped newlines and quotes to fit json format")
    fact_name: str = Field(
        description="fact name to save code execution result, should be unique, lowercase, snake_case, without spaces and special characters"
    )
    facts: dict | None = None


################### Observations ###################


class GaiaQuestion(GaiaObservation):
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


class SearchResultsObservation(GaiaObservation):
    kind: Literal["search_results_observation"] = "search_results_observation"
    query: str
    serp: list[dict[str, str]]


class PageObservation(GaiaObservation):
    kind: Literal["page_observation"] = "page_observation"
    text: str
    current_page: int
    total_pages: int
    error: int | None = None


class CalculationResultObservation(GaiaObservation):
    kind: Literal["calculation_result_observation"] = "calculation_result_observation"
    name: str
    result: str


class CodeResultObservation(GaiaObservation):
    kind: Literal["code_result_observation"] = "code_result_observation"
    name: str
    result: str
    stdout: str
    stderr: str


class PreviousFactsObservation(GaiaObservation):
    kind: Literal["previous_facts_observation"] = "previous_facts_observation"
    reasoning: str = "These facts were gathered in previous steps that were trimmed to fit the context size limit"
    facts: dict[str, Any]


class GaiaAnswer(GaiaAction, StopStep):
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


class ActionExecutionFailure(GaiaObservation, Error):
    kind: Literal["action_execution_failure"] = "action_execution_failure"
    error: str


GaiaStep = Union[
    PlanThought,
    ListOfFactsThought,
    SourcesThought,
    DraftPlansThought,
    ReadingResultThought,
    NewFactThought,
    ReasoningThought,
    StartSubtask,
    FinishSubtask,
    SearchAction,
    ReadDocumentAction,
    NextPageAction,
    ConvertFactAction,
    # UseCalculatorAction,
    PythonCodeAction,
    GaiaQuestion,
    SearchResultsObservation,
    PageObservation,
    # CalculationResultObservation,
    CodeResultObservation,
    PreviousFactsObservation,
    GaiaAnswer,
    ActionExecutionFailure,
    LLMOutputParsingFailureAction,
    SetNextNode,
    ImageObservation,
]

GaiaAgentStep: TypeAlias = Annotated[
    Union[
        # thoughts
        PlanThought,
        ListOfFactsThought,
        SourcesThought,
        DraftPlansThought,
        ReadingResultThought,
        NewFactThought,
        ReasoningThought,
        StartSubtask,
        FinishSubtask,
        # actions
        SearchAction,
        ReadDocumentAction,
        NextPageAction,
        ConvertFactAction,
        # UseCalculatorAction,
        PythonCodeAction,
        GaiaAnswer,
    ],
    Field(discriminator="kind"),
]

actions = [
    SearchAction,
    ReadDocumentAction,
    NextPageAction,
    ConvertFactAction,
    # UseCalculatorAction,
    PythonCodeAction,
    GaiaAnswer,
]


def get_allowed_steps(subtasks: bool, plan_thoughts: bool) -> str:
    if plan_thoughts:
        steps = Union[PlanThought, ListOfFactsThought, DraftPlansThought, SourcesThought]
    else:
        if subtasks:
            steps = Union[
                ReadingResultThought,
                NewFactThought,
                ReasoningThought,
                SearchAction,
                ReadDocumentAction,
                NextPageAction,
                ConvertFactAction,
                # UseCalculatorAction,
                PythonCodeAction,
                GaiaAnswer,
                StartSubtask,
                FinishSubtask,
            ]
        else:
            steps = Union[
                ReadingResultThought,
                NewFactThought,
                ReasoningThought,
                SearchAction,
                ReadDocumentAction,
                NextPageAction,
                ConvertFactAction,
                # UseCalculatorAction,
                PythonCodeAction,
                GaiaAnswer,
            ]
    steps_alias = Annotated[steps, Field(discriminator="kind")]
    return get_step_schemas_from_union_type(steps_alias)

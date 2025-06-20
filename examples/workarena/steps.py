from typing import Annotated, Literal, Union

from pydantic import Field

from tapeagents.core import FinalObservation, LLMOutputParsingFailureAction, SetNextNode, StopStep, Tape, Thought
from tapeagents.dialog_tape import DialogContext, UserStep
from tapeagents.steps import ActionExecutionFailure, ReasoningThought
from tapeagents.tools.browser import (
    ClickBIDAction,
    GoBackAction,
    GoForwardAction,
    HoverAction,
    InputTextAction,
    PageDownAction,
    PageObservation,
    PageUpAction,
    PressAction,
    SelectOptionAction,
)


class ReflectionThought(Thought):
    """
    Review the current state of the page and previous steps to find the best possible next action to accomplish the task:
    1. Produce reasoning thougt explaining the observed state, think about which blocks could be relevant to the given task and its current state, note relevant BIDs.
    2. Produce list of things to do to accomplish the task. Mention all the fields that need to be filled, buttons that need to be clicked, etc.
    3. Reflect if you last action succes, check if it produced the expected effect on the page.
    4. Reflect on you last actions to check if you are being stuck with repeating the same action. If you repeatedly failed to perform the same action with the page element, propose a new way of interacting.
    5. Describe the next action to be performed and expected effect on the page.
    """

    kind: Literal["reflection_thought"] = "reflection_thought"  # type: ignore
    page_state: str = Field(
        description="description of the current page state, think about which blocks could be relevant to the given task and its current state, note relevant BIDs"
    )
    last_action_kind: str = Field(description="describe the kind of the last action")
    last_action_achieved_effect: str = Field(
        description="describe the the changes in the page state achieved by the last action that you can observe"
    )
    last_action_expected_effect: str = Field(
        description="describe the expected changes in the page state that should be achieved by the last action"
    )
    last_action_success: bool = Field(
        description="success of the last action, True if it produced the expected effect on the page"
    )
    last_action_repetitions: str = Field(
        description="check if you are being stuck with repeating the same action and describe here"
    )
    are_we_stuck: bool = Field(description="if you are being stuck with repeating the same action")
    task_solved: bool = Field(description="if the main task given by used is solved")
    todo_list: list[str] = Field(
        description="detailed list of things to do to accomplish the task, down to the fields level"
    )
    next_action: str = Field(description="describe the next action to be performed and expected effect on the page")
    page_line: str = Field(
        description="verbatim quote of the line of the page that contains required element. Empty if next action is not related to specific page element"
    )


class FinalAnswerAction(StopStep):
    """
    Action that provides the final answer to the user after completing the task.
    Should be produced when the agent has finished the task.
    When the task has question about numerical value, the answer should contain only one number!
    """

    kind: Literal["final_answer_action"] = "final_answer_action"  # type: ignore
    text: str = Field(description="final answer to the user")


WorkArenaStep = Union[
    UserStep,
    PageObservation,
    ActionExecutionFailure,
    LLMOutputParsingFailureAction,
    # thoughts
    ReasoningThought,
    ReflectionThought,
    # browser actions
    ClickBIDAction,
    SelectOptionAction,
    HoverAction,
    InputTextAction,
    PressAction,
    PageDownAction,
    PageUpAction,
    # TabFocusAction,
    # NewTabAction,
    # CloseTabAction,
    GoBackAction,
    GoForwardAction,
    # GotoPageAction,
    FinalAnswerAction,
    SetNextNode,
]


class WorkArenaTape(Tape[DialogContext, WorkArenaStep]):
    pass


WorkArenaAgentStep = (
    # thoughts
    ReasoningThought,
    ReflectionThought,
    # browser actions
    ClickBIDAction,
    SelectOptionAction,
    HoverAction,
    InputTextAction,
    PressAction,
    PageDownAction,
    PageUpAction,
    # TabFocusAction,
    # NewTabAction,
    # CloseTabAction,
    GoBackAction,
    GoForwardAction,
    # GotoPageAction,
    FinalAnswerAction,
    FinalObservation,
)


steps_cls = Annotated[Union[WorkArenaStep], Field(discriminator="kind")]

import json

from litellm.utils import ChatCompletionMessageToolCall
from pydantic import TypeAdapter

from tapeagents.core import Prompt, Tape
from tapeagents.dialog_tape import (
    AnnotationAction,
    AnnotatorFreeFormThought,
    AssistantStep,
    DialogAnnotator,
    DialogAnnotatorTape,
    DialogContext,
    DialogTape,
    UserStep,
)
from tapeagents.llms import LiteLLM, LLMStream
from tapeagents.renderers import render_dialog_plain_text
from tapeagents.tool_calling import ToolCalls, ToolResult, ToolSpec

_ANNOTATOR_PROMPT: str = """Here is a dialog between a user and an assistant.

{dialog}

You will judge whether everything that the assistant said
IN THEIR LAST MESSAGE is supported by the result of the function calls.

You will output as JSON of the following format:

{{
    "thinking": <your thought process>
    "answer": <output "yes" or "no">
}}

You must output only the JSON and nothing else. Go!
"""


class GroundednessAnnotator(DialogAnnotator):
    """
    Annotates the steps of a given dialog tape based on the results of function calls in the tape.
    Produces the binary correctness of the assistant's last message.
    """

    def make_prompt(self, tape: DialogAnnotatorTape):
        """
        Creates a prompt for the annotator based on the dialog tape.

        Args:
            tape (DialogAnnotatorTape): The dialog tape to be annotated.

        Returns:
            Prompt: The prompt to be used by the annotator.
        """
        return Prompt(
            messages=[
                {"role": "user", "content": _ANNOTATOR_PROMPT.format(dialog=render_dialog_plain_text(tape.context))}
            ]
        )

    def generate_steps(self, tape: Tape, llm_stream: LLMStream):
        """
        Generates annotation steps based on the LLM stream output.

        Args:
            tape: tape with previous steps.
            llm_stream (LLMStream): The stream of text from the language model.

        Yields:
            AnnotatorFreeFormThought: The thought process of the annotator.
            AnnotationAction: The annotation action indicating whether the assistant's response is grounded.

        Raises:
            ValueError: If the answer in the LLM stream output is not "yes" or "no".
        """
        text = llm_stream.get_text()
        result = json.loads(text)
        yield AnnotatorFreeFormThought(content=result["thinking"])
        if result["answer"] == "yes":
            yield AnnotationAction(annotation=dict(grounded=True))
        elif result["answer"] == "no":
            yield AnnotationAction(annotation=dict(grounded=False))
        else:
            raise ValueError("Invalid answer")


TOOL_SCHEMAS = TypeAdapter(list[ToolSpec]).validate_python(
    [
        {
            "type": "function",
            "function": {
                "name": "search_web",
                "description": "Get some information from the web search",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "The web search query"},
                    },
                    "required": ["query"],
                },
            },
        }
    ]
)


def main():
    dialog1 = DialogTape(
        context=DialogContext(tools=TOOL_SCHEMAS),
        steps=[
            UserStep(content="Tell me about Montreal"),
            ToolCalls(
                tool_calls=[
                    ChatCompletionMessageToolCall(
                        function={"name": "search_web", "arguments": """{"query": "facts about Montreal"}"""}
                    )
                ]
            ),
            ToolResult(content="Montreal is a city in Canada", tool_call_id="1"),
            AssistantStep(content="Montreal is a city in Canada. It is 123 years old."),
        ],
    )

    llm = LiteLLM(model_name="gpt-4o-mini-2024-07-18")
    annotator = GroundednessAnnotator.create(llm)

    # Annotate the dialog
    annotator_tape1 = annotator.annotate(dialog1)
    print(f"Tape:\n{annotator_tape1.model_dump_json(indent=2)}")
    print(f"Annotation:\n{annotator.get_annotation(annotator_tape1)}")

    # Let's replace the last step and re-annotate
    dialog2 = dialog1.model_copy(
        update=dict(steps=dialog1.steps[:-1] + [AssistantStep(content="Montreal is a city in Canada.")])
    )
    annotator_tape2 = annotator.annotate(dialog2)
    print(f"\n\nTape:\n{annotator_tape2.model_dump_json(indent=2)}")
    print(f"Annotation:\n{annotator.get_annotation(annotator_tape2)}")


if __name__ == "__main__":
    main()

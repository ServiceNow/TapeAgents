import json

from litellm.utils import ChatCompletionMessageToolCall
from pydantic import TypeAdapter

from tapeagents.core import Prompt
from tapeagents.dialog_tape import (
    AnnotationAction,
    AnnotatorFreeFormThought,
    AssistantStep,
    DialogTape,
    DialogAnnotator,
    DialogAnnotatorTape,
    DialogContext,
    ToolCalls,
    ToolResult,
    ToolSpec,
    UserStep,
)
from tapeagents.llms import LiteLLM, LLMStream
from tapeagents.rendering import render_dialog_plain_text

_ANNOTATOR_PROMPT = """Here is a dialog between a user and an assistant. 

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
    def make_prompt(self, tape: DialogAnnotatorTape):
        return Prompt(
            messages=[
                {"role": "user", "content": _ANNOTATOR_PROMPT.format(dialog=render_dialog_plain_text(tape.context))}
            ]
        )

    def generate_steps(self, _, llm_stream: LLMStream):
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


def try_annotator():
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
    dialog2 = dialog1.model_copy(
        update=dict(steps=dialog1.steps[:-1] + [AssistantStep(content="Montreal is a city in Canada.")])
    )

    annotator = GroundednessAnnotator.create(LiteLLM(model_name="gpt-3.5-turbo"))
    annotator_tape1 = annotator.annotate(dialog1)
    print(json.dumps(annotator_tape1.model_dump(), indent=2))
    print(annotator.get_annotation(annotator_tape1))

    annotator_tape2 = annotator.annotate(dialog2)
    print(json.dumps(annotator_tape2.model_dump(), indent=2))
    print(annotator.get_annotation(annotator_tape2))


if __name__ == "__main__":
    try_annotator()

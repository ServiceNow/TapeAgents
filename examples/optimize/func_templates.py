from tapeagents.dialog_tape import ToolResult
from tapeagents.llm_function import InputVar, LLMFunctionTemplate, AssistantOutput, RationaleOutput, ToolCallOutput


def render_contexts(contexts: list[str]) -> str:
    if not contexts:
        return "N/A"
    return "\n".join(f"[{i + 1}] «{t}»" for i, t in enumerate(contexts))


class ContextInput(InputVar):
    def render(self, step: ToolResult):
        return render_contexts(step.content)


def make_answer_template() -> LLMFunctionTemplate:
    return LLMFunctionTemplate(
        desc="Answer questions with short factoid answers.",
        inputs=[
            ContextInput(name="context", desc="may contain relevant facts", separator="\n"),
            InputVar(name="question"),
        ],
        outputs=[
            RationaleOutput.for_output("answer"),
            AssistantOutput(name="answer", desc="often between 1 and 5 words")
        ]
    )        
    
    
def make_query_template() -> LLMFunctionTemplate:
    return LLMFunctionTemplate(
        desc="Write a simple search query that will help answer a complex question.",
        inputs=[
            ContextInput(name="context", desc="may contain relevant facts", separator="\n"),
            InputVar(name="question"),
        ],
        outputs=[
            RationaleOutput.for_output("query"),
            ToolCallOutput(name="query", tool_name="retrieve", arg_name="query")
        ]
    )       
import json
import pathlib

from tapeagents.dialog_tape import (
    AssistantStep,
    AssistantThought,
    UserStep,
)
from tapeagents.tool_calling import FunctionCall, ToolCall, ToolCalls, ToolResult

res_dir = pathlib.Path(__file__).parent.resolve() / "res"


def load_rag_demos() -> tuple[list, list]:
    with open(res_dir / "llm_function_rag_demos.json") as f:
        demos_json = json.load(f)
    partial_demos = []
    demos = []
    for demo in demos_json:
        if demo.get("augmented"):
            demo = {
                "question": UserStep(content=demo["question"]),
                "context": ToolResult(content=demo["context"], tool_call_id=""),
                "reasoning": AssistantThought(content=demo["reasoning"]),
                "answer": AssistantStep(content=demo["answer"]),
            }
            demos.append(demo)
        else:
            demo = {
                "question": UserStep(content=demo["question"]),
                "answer": AssistantStep(content=demo["answer"]),
            }
            partial_demos.append(demo)
    return partial_demos, demos


def load_agentic_rag_demos() -> dict[str, tuple[list, list]]:
    """Loads full demos only"""
    with open(res_dir / "agentic_rag_demos.json") as f:
        demos_json = json.load(f)
    result = {}
    for predictor, predictor_demos in demos_json.items():
        predictor_demos = [d for d in predictor_demos if d.get("augmented")]
        demos = []
        if "query" in predictor:
            for demo in predictor_demos:
                tc = ToolCall(function=FunctionCall(name="retrieve", arguments={"query": demo["query"]}))
                demo = {
                    "question": UserStep(content=demo["question"]),
                    "context": ToolResult(content=demo["context"]),
                    "reasoning": AssistantThought(content=demo["reasoning"]),
                    "query": ToolCalls(tool_calls=[tc]),
                }
                demos.append(demo)
            result[f"query{predictor[-2]}"] = ([], demos)
        elif predictor == "generate_answer":
            for demo in predictor_demos:
                demo = {
                    "question": UserStep(content=demo["question"]),
                    "context": ToolResult(content=demo["context"], tool_call_id=""),
                    "reasoning": AssistantThought(content=demo["reasoning"]),
                    "answer": AssistantStep(content=demo["answer"]),
                }
                demos.append(demo)
            result["answer"] = ([], demos)
        else:
            raise ValueError(f"Unknown predictor {predictor}")
    return result

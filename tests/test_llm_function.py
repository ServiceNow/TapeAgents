import sys
from pathlib import Path

from tapeagents.core import Prompt
from tapeagents.dialog_tape import UserStep
from tapeagents.llm_function import (
    AssistantOutput,
    Input,
    LLMFunctionTemplate,
    ReasoningOutput,
)
from tapeagents.llms import LLMEvent, LLMOutput, LLMStream
from tapeagents.tool_calling import ToolResult
from tapeagents.utils import diff_strings

sys.path.append(str(Path(__file__).parent.parent.resolve()))

from examples.optimize.func_templates import make_answer_template, make_query_template
from examples.optimize.load_demos import load_agentic_rag_demos, load_rag_demos

TEST_INPUT_STEP1 = UserStep(
    content="What is the nationality of the chef and restaurateur featured in Restaurant: Impossible?"
)
TEST_INPUT_STEP2 = UserStep(content="What castle did David Gregory inherit?")
TEST_INPUT_STEP3 = UserStep(content="How many storeys are in the castle that David Gregory inherited?")
TEST_CONTEXT_STEP2 = ToolResult(
    content=[
        """David Gregory (physician) | David Gregory (20 December 1625 – 1720) was a Scottish physician and inventor. His surname is sometimes spelt as Gregorie, the original Scottish spelling. He inherited Kinnairdy Castle in 1664. Three of his twenty-nine children became mathematics professors. He is credited with inventing a military cannon that Isaac Newton described as "being destructive to the human species". Copies and details of the model no longer exist. Gregory's use of a barometer to predict farming-related weather conditions led him to be accused of witchcraft by Presbyterian ministers from Aberdeen, although he was never convicted.""",
        """Gregory Tarchaneiotes | Gregory Tarchaneiotes (Greek: Γρηγόριος Ταρχανειώτης , Italian: "Gregorio Tracanioto" or "Tracamoto" ) was a "protospatharius" and the long-reigning catepan of Italy from 998 to 1006. In December 999, and again on February 2, 1002, he reinstituted and confirmed the possessions of the abbey and monks of Monte Cassino in Ascoli. In 1004, he fortified and expanded the castle of Dragonara on the Fortore. He gave it three circular towers and one square one. He also strengthened Lucera.""",
        """David Gregory (mathematician) | David Gregory (originally spelt Gregorie) FRS (? 1659 – 10 October 1708) was a Scottish mathematician and astronomer. He was professor of mathematics at the University of Edinburgh, Savilian Professor of Astronomy at the University of Oxford, and a commentator on Isaac Newton's "Principia".""",
    ],
    tool_call_id="",
)
TEST_CONTEXT_STEP3 = ToolResult(
    content=[
        """David Gregory (physician) | David Gregory (20 December 1625 – 1720) was a Scottish physician and inventor. His surname is sometimes spelt as Gregorie, the original Scottish spelling. He inherited Kinnairdy Castle in 1664. Three of his twenty-nine children became mathematics professors. He is credited with inventing a military cannon that Isaac Newton described as "being destructive to the human species". Copies and details of the model no longer exist. Gregory's use of a barometer to predict farming-related weather conditions led him to be accused of witchcraft by Presbyterian ministers from Aberdeen, although he was never convicted.""",
        """David Webster (architect) | David Webster (1885–1952) was a Scottish-Canadian architect best known for his designs of elementary schools in Saskatoon, Saskatchewan, Canada. His school designs were often in a Collegiate Gothic style emphasizing a central tower, locally referred to as a "castle style". Along with other local architects of his era, such as Walter LaChance and Storey and Van Egmond, Webster prospered during the province’s 1912 economic boom which sparked a frenzy of new construction.""",
        """David S. Castle | David S. Castle (13 February 1884 – 28 October 1956) was an architect in Texas.""",
    ]
)

res_path = Path(__file__).parent.resolve() / ".." / "tests" / "res"
examples_res_path = Path(__file__).parent.resolve() / ".." / "examples" / "res"


def remove_empty_lines(text: str) -> str:
    return "\n".join(filter(lambda x: x.strip(), text.split("\n")))


def test_dspy_qa_prompt():
    func = LLMFunctionTemplate(
        desc="Answer questions with short factoid answers.",
        inputs=[Input(name="question")],
        outputs=[AssistantOutput(name="answer", desc="often between 1 and 5 words")],
    )

    render = func.make_prompt([TEST_INPUT_STEP1]).messages[0]["content"]
    with open(res_path / "llm_function" / "qa.txt", "r") as f:
        gold = f.read()
    if render != gold:
        print(diff_strings(render, gold))
        assert False


def test_dspy_cot_prompt():
    func = LLMFunctionTemplate(
        desc="Answer questions with short factoid answers.",
        inputs=[Input(name="question")],
        outputs=[
            ReasoningOutput.for_output("answer"),
            AssistantOutput(name="answer", desc="often between 1 and 5 words"),
        ],
    )

    render = func.make_prompt([TEST_INPUT_STEP1]).messages[0]["content"]
    with open(res_path / "llm_function" / "cot.txt", "r") as f:
        gold = f.read()
    if render != gold:
        print(diff_strings(render, gold))
        assert False


def test_fewshot_prompt():
    func = make_answer_template()
    partial_demos, demos = load_rag_demos()
    func.partial_demos = partial_demos
    func.demos = demos

    render = func.make_prompt([TEST_CONTEXT_STEP2, TEST_INPUT_STEP2]).messages[0]["content"]
    with open(res_path / "llm_function" / "rag.txt", "r") as f:
        gold = f.read()

    render = remove_empty_lines(render)
    gold = remove_empty_lines(gold)
    if render != gold:
        print(diff_strings(render, gold))
        assert False


def test_query_prompt():
    func = make_query_template()
    _, demos = load_agentic_rag_demos()["query1"]
    func.demos = demos

    render = func.make_prompt([TEST_CONTEXT_STEP3, TEST_INPUT_STEP3]).messages[0]["content"]
    with open(res_path / "llm_function" / "query.txt", "r") as f:
        gold = f.read()

    render = remove_empty_lines(render)
    gold = remove_empty_lines(gold)
    if render != gold:
        print(diff_strings(render, gold))
        assert False


def test_generate_steps_answer_template():
    ## gpt-3.5-turbo
    llm_output = "Answer: Anakin Skywalker is Leia's father."
    func = make_answer_template()
    steps = list(func.generate_steps(llm_stream=_llm_sream(llm_output), agent=None, tape=None))
    assert steps[0].kind == "assistant_thought"
    assert steps[0].content == ""
    assert steps[1].kind == "assistant"
    assert steps[1].content == "Anakin Skywalker is Leia's father."

    llm_output = "determine the familial connection. Anakin Skywalker is the father of Luke Skywalker and Leia Organa. Therefore, he is Leia's father.  \nAnswer: Father"
    steps = list(func.generate_steps(llm_stream=_llm_sream(llm_output), agent=None, tape=None))
    assert steps[0].kind == "assistant_thought"
    assert (
        steps[0].content
        == "determine the familial connection. Anakin Skywalker is the father of Luke Skywalker and Leia Organa. Therefore, he is Leia's father."
    )
    assert steps[1].kind == "assistant"
    assert steps[1].content == "Father"


def test_generate_steps_query_template():
    # gpt-3.5-turbo
    llm_output = ' produce the query. We know that Anakin Skywalker is the father of Princess Leia. Therefore, we need to search for the relationship between a father and daughter.\nQuery: "relationship between father and daughter" site:wikipedia.org'

    func = make_query_template()
    steps = list(func.generate_steps(llm_stream=_llm_sream(llm_output), agent=None, tape=None))
    assert steps[0].kind == "assistant_thought"
    assert (
        steps[0].content
        == "produce the query. We know that Anakin Skywalker is the father of Princess Leia. Therefore, we need to search for the relationship between a father and daughter."
    )
    assert steps[1].kind == "assistant"
    assert steps[1].tool_calls[0].function.name == "retrieve"
    assert (
        steps[1].tool_calls[0].function.arguments["query"]
        == '"relationship between father and daughter" site:wikipedia.org'
    )

    # gpt-4o-mini
    llm_output = "Context: N/A  \nQuestion: What relation is Anakin Skywalker to Princess Leia?  \nReasoning: Let's think step by step in order to identify the familial connections between Anakin Skywalker and Princess Leia. We need to find information about Anakin Skywalker’s identity, his family, and how that relates to Princess Leia. We will search for details about Anakin's lineage and any connections to Leia.  \nQuery: \"Anakin Skywalker family relation to Princess Leia\""

    steps = list(func.generate_steps(llm_stream=_llm_sream(llm_output), agent=None, tape=None))
    assert steps[0].kind == "assistant_thought"
    assert (
        steps[0].content
        == "identify the familial connections between Anakin Skywalker and Princess Leia. We need to find information about Anakin Skywalker’s identity, his family, and how that relates to Princess Leia. We will search for details about Anakin's lineage and any connections to Leia."
    )
    assert steps[1].kind == "assistant"
    assert steps[1].tool_calls[0].function.name == "retrieve"
    assert steps[1].tool_calls[0].function.arguments["query"] == '"Anakin Skywalker family relation to Princess Leia"'


def _llm_sream(output: str):
    def _generator(output: str):
        yield LLMEvent(output=LLMOutput(content=output))

    return LLMStream(_generator(output), Prompt())


if __name__ == "__main__":
    test_dspy_qa_prompt()
    test_dspy_cot_prompt()
    test_fewshot_prompt()
    test_query_prompt()
    test_generate_steps_answer_template()
    test_generate_steps_query_template()
    test_generate_steps_query_template()

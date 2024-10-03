import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.resolve()))

from examples.optimize import load_agentic_rag_demos, load_rag_demos, make_answer_template, make_query_template, render_contexts
from tapeagents.dialog_tape import AssistantStep, AssistantThought, ToolResult, UserStep
from tapeagents.llm_function import InputStep, LLMFunctionTemplate, OutputStep
from tapeagents.utils import diff_strings


TEST_INPUT_STEP1 = UserStep(content="What is the nationality of the chef and restaurateur featured in Restaurant: Impossible?")    
TEST_INPUT_STEP2 = UserStep(content="What castle did David Gregory inherit?")
TEST_INPUT_STEP3 = UserStep(content="How many storeys are in the castle that David Gregory inherited?")
TEST_CONTEXT_STEP2 = ToolResult(
    content=[
        """David Gregory (physician) | David Gregory (20 December 1625 – 1720) was a Scottish physician and inventor. His surname is sometimes spelt as Gregorie, the original Scottish spelling. He inherited Kinnairdy Castle in 1664. Three of his twenty-nine children became mathematics professors. He is credited with inventing a military cannon that Isaac Newton described as "being destructive to the human species". Copies and details of the model no longer exist. Gregory's use of a barometer to predict farming-related weather conditions led him to be accused of witchcraft by Presbyterian ministers from Aberdeen, although he was never convicted.""",
        """Gregory Tarchaneiotes | Gregory Tarchaneiotes (Greek: Γρηγόριος Ταρχανειώτης , Italian: "Gregorio Tracanioto" or "Tracamoto" ) was a "protospatharius" and the long-reigning catepan of Italy from 998 to 1006. In December 999, and again on February 2, 1002, he reinstituted and confirmed the possessions of the abbey and monks of Monte Cassino in Ascoli. In 1004, he fortified and expanded the castle of Dragonara on the Fortore. He gave it three circular towers and one square one. He also strengthened Lucera.""",
        """David Gregory (mathematician) | David Gregory (originally spelt Gregorie) FRS (? 1659 – 10 October 1708) was a Scottish mathematician and astronomer. He was professor of mathematics at the University of Edinburgh, Savilian Professor of Astronomy at the University of Oxford, and a commentator on Isaac Newton's "Principia"."""
    ],
    tool_call_id=""
)
TEST_CONTEXT_STEP3 = ToolResult(
    content=[
        """David Gregory (physician) | David Gregory (20 December 1625 – 1720) was a Scottish physician and inventor. His surname is sometimes spelt as Gregorie, the original Scottish spelling. He inherited Kinnairdy Castle in 1664. Three of his twenty-nine children became mathematics professors. He is credited with inventing a military cannon that Isaac Newton described as "being destructive to the human species". Copies and details of the model no longer exist. Gregory's use of a barometer to predict farming-related weather conditions led him to be accused of witchcraft by Presbyterian ministers from Aberdeen, although he was never convicted.""",
        """David Webster (architect) | David Webster (1885–1952) was a Scottish-Canadian architect best known for his designs of elementary schools in Saskatoon, Saskatchewan, Canada. His school designs were often in a Collegiate Gothic style emphasizing a central tower, locally referred to as a "castle style". Along with other local architects of his era, such as Walter LaChance and Storey and Van Egmond, Webster prospered during the province’s 1912 economic boom which sparked a frenzy of new construction.""",
        """David S. Castle | David S. Castle (13 February 1884 – 28 October 1956) was an architect in Texas."""
    ]
)
    
res_path = Path(__file__).parent.resolve() / ".." / "tests" / "res"
examples_res_path = Path(__file__).parent.resolve() / ".." / "examples" / "res"
    
def remove_empty_lines(text: str) -> str:
    return "\n".join(filter(lambda x: x.strip(), text.split("\n")))    
    
def test_dspy_qa_prompt():
    func = LLMFunctionTemplate(
        desc="Answer questions with short factoid answers.",
        inputs=[
            InputStep(name="question")
        ],
        outputs=[
            OutputStep(name="answer", desc="often between 1 and 5 words")
        ]
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
        inputs=[
            InputStep(name="question")
        ],
        outputs=[
            OutputStep(prefix="Reasoning: Let's think step by step in order to", desc="${produce the answer}. We ..."),
            OutputStep(name="answer", desc="often between 1 and 5 words")
        ]
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
# https://github.com/deepseek-ai/DeepSeek-Math/blob/b8b0f8ce093d80bf8e9a641e44142f06d092c305/evaluation/data_processing/process_utils.py
import regex

from examples.rl_gsm8k.deepseek_math_eval.answer_extraction import (
    extract_math_answer, strip_string)
from examples.rl_gsm8k.deepseek_math_eval.eval_utils import parse_ground_truth


def process_eurus_test(item):
    if "ability" not in item:
        # math 500 test set
        answer = [item["expected_answer"]]
        return {
            "dataset": "math500",
            # Same prompt as https://github.com/PRIME-RL/PRIME/blob/49a58a8e4afd464f559f8d9f80418052f29cf3e4/README.md?plain=1#L93
            "task": item["problem"] + "\n\nPresent the answer in LaTex format: \\boxed{Your answer}",
            "answer": answer
        }
    else:
        # Eurus train set
        if item["ability"] != "math":
            return None
        answer = item["reward_model"]["ground_truth"]
        # format matrices
        # remove new lines
        answer = answer.replace("\n", "")
        answer = "\\boxed{" + answer + "}"
        answer = extract_math_answer(item["prompt"][1]["content"], answer, task="cot")
        return {
            "dataset": item["data_source"],
            "task": item["prompt"][1]["content"],
            "answer": answer
        }
    

def process_gsm8k_test(item):
    _, answer = parse_ground_truth(item, "gsm8k")
    sample = {"dataset": "gsm8k-cot", "task": item["question"], "answer": answer}
    return sample


def process_math_test(item):
    question = item["problem"]
    try:
        answer = extract_math_answer(question, item["solution"], task="cot")
    except Exception:
        return
    sample = {"dataset": "math-cot", "level": item.get("level", ""), "type": item.get("type", ""), "task": question, "answer": answer}
    return sample


def process_math_sat(item):
    options = item["options"].strip()
    assert "A" == options[0]
    options = "(" + options
    for ch in "BCDEFG":
        if f" {ch}) " in options:
            options = regex.sub(f" {ch}\) ", f" ({ch}) ", options)
    question = f"{item['question'].strip()}\nWhat of the following is the right choice? Explain your answer.\n{options.strip()}"
    messages = [{"role": "user", "content": question}, {"role": "assistant", "content": item["Answer"]}]
    item = {
        "dataset": "math_sat",
        "id": item["id"],
        "language": "en",
        "messages": messages,
        "answer": item["Answer"],
    }
    yield item


def process_ocwcourses(item):
    messages = [
        {"role": "user", "content": item["problem"].strip()},
        {"role": "assistant", "content": item["solution"].strip()},
    ]
    item = {"dataset": "OCWCourses", "id": item["id"], "language": "en", "messages": messages, "answer": item["answer"]}
    yield item


def process_mmlu_stem(item):
    options = item["options"]
    for i, (label, option) in enumerate(zip("ABCD", options)):
        options[i] = f"({label}) {str(option).strip()}"
    options = ", ".join(options)
    question = f"{item['question'].strip()}\nWhat of the following is the right choice? Explain your answer.\n{options}"
    messages = [{"role": "user", "content": question}, {"role": "assistant", "content": item["answer"]}]
    item = {"dataset": "MMLU-STEM", "id": item["id"], "language": "en", "messages": messages, "answer": item["answer"]}
    yield item


def process_mgsm_zh(item):
    item["answer"] = item["answer"].replace(",", "")
    yield item


def process_cmath(item):
    item = {
        "dataset": "cmath",
        "id": item["id"],
        "grade": item["grade"],
        "reasoning_step": item["reasoning_step"],
        "messages": [{"role": "user", "content": item["question"].strip()}, {"role": "assistant", "content": ""}],
        "answer": item["golden"].strip().replace(",", ""),
    }
    yield item


def process_agieval_gaokao_math_cloze(item):
    item = {
        "dataset": "agieval-gaokao-math-cloze",
        "id": item["id"],
        "messages": [{"role": "user", "content": item["question"].strip()}, {"role": "assistant", "content": ""}],
        "answer": [strip_string(ans) for ans in item["answer"].strip().split(";")],
    }
    yield item


def process_agieval_gaokao_mathqa(item):
    question = item["question"].strip()
    options = []
    for option in item["options"]:
        option = option.strip()
        assert option[0] == "("
        assert option[2] == ")"
        assert option[1] in "ABCD"
        option = f"{option[1]}: {option[3:].strip()}"
        options.append(option.strip())
    question = f"{question}\n{options}"
    item = {
        "dataset": "agieval-gaokao-mathqa",
        "id": item["id"],
        "messages": [{"role": "user", "content": question}, {"role": "assistant", "content": ""}],
        "answer": item["label"],
    }
    yield item


def process_agieval_gaokao_mathqa_few_shot_cot_test(item):
    question = item["question"].strip().rstrip("\\")
    options = " ".join([opt.strip() for opt in item["options"]])
    question = f"{question}\n从以下选项中选择:    {options}"
    item = {
        "dataset": "agieval-gaokao-mathqa",
        "id": item["id"],
        "messages": [{"role": "user", "content": question}, {"role": "assistant", "content": ""}],
        "answer": item["label"],
    }
    yield item


def process_minif2f_isabelle(item):
    question = f"(*### Problem\n\n{item['informal_statement'].strip()}\n\n### Solution\n\n{item['informal_proof'].strip()} *)\n\nFormal:\n{item['formal_statement'].strip()}"
    item = {
        "dataset": "minif2f-isabelle",
        "id": item["id"],
        "messages": [{"role": "user", "content": question}, {"role": "assistant", "content": ""}],
        "answer": "placeholder",
    }
    yield item

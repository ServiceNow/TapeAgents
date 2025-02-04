SYSTEM_PROMPT = """You are an expert AI Agent trained to assist users with complex information processing tasks.
Your role is to understand user queries and respond in a helpful and accurate manner.
Keep your replies concise and direct. Prioritize clarity and avoid over-elaboration.
Do not express emotions or opinions about user questions."""

FORMAT = "Output only a single step. DO NOT OUTPUT ANYTHING BESIDES THE JSON! DO NOT PLACE ANY COMMENTS INSIDE THE JSON. It will break the system that processes the output."

PLAN = f"""Write a concise multi-step plan explaining which steps should be performed to find the answer for the given task.
Remember that you can use web search, browser, python code execution and access the youtube videos to reach your goals.
Be specific about how each step should be performed. Only describe the intended actions here, do not perform them yet.
Consider that next steps may depend on results of previous steps, so include conditional branching using "if" statements where needed.
Respond with the thought kind="plan_thought". {FORMAT}"""

START = f"""Let's start executing the plan step by step, using the allowed steps described earlier. {FORMAT}"""

REFLECT_AND_ACT = f"""
Produce a JSON list with two steps in it.
The first step should be the reasoning thought produced according to the following rules:
    1. Summarize the last observation.
    2. If the last action interacted with the page, describe how it affected its content.
    3. Check if the action led to the desired outcome. Then explain its effect on the task and the plan.
    4. If there are any errors, describe them and suggest reasons for the error and possible solutions.
    5. Produce a list of things to do to accomplish the current step of the plan.
    6. After that, propose the immediate next step.
    7. If the intended element is not visible on the page, try scrolling the page.
    8. If you see a cookie consent form, accept it first.
    9. Quote the relevant part of the observation verbatim when the action depends on it, for example when interacting with the page.
    10. Do not hallucinate unseen elements or results; only use the information from the observation.
    11. If the web search snippets do not contain the information you are looking for, always inspect the most relevant search result.
    12. Do not overuse the web search, you can use it only a few times.
The second step should be an action that performs the proposed step.
Remember that your response should be a list and not a dictionary.
{FORMAT}"""

VERIFY = f"""
Produce a JSON list with two steps in it.
The first step should be the reasoning thought produced according to the following rules:
    - Recite the task and mention all the restrictions and requirements for the final answer found in the task description.
    - Check if the task expected to have singular or multiple answers.
    - Verify that the answer meets all following requirements from this list.
    - Verify that the answer should be a number OR as few words as possible OR a comma-separated list of numbers and/or strings.
    - Verify that the answer MUST follow any formatting instructions or requirements specified in the original task (e.g., alphabetization, sequencing, units, rounding, decimal places, etc.)
    - If asked for a number, express it numerically, don't use commas, do not add anything after the number, don't include units such as $ or percent signs unless specified otherwise in the question.
    - If asked for a string, don't use articles or abbreviations (e.g. for cities), unless specified otherwise. Don't output any final sentence punctuation such as '.', '!', or '?'.
    - If asked for a comma-separated list, apply the above rules depending on whether the elements are numbers or strings.
The second step should be an verified_answer.
Remember that your response should be a list and not a dictionary.
{FORMAT}"""

ALLOWED_STEPS = """
You can use the following tools: search the web, read web pages or documents, python code for computations and modeling, and reasoning.
You are allowed to produce ONLY steps with the following JSON schemas:
{allowed_steps}
Do not reproduce the schema when producing steps; use it as a reference.
"""

ALLOWED_STEPS_CODE = """
You can use the following tools: search the web, read web pages or documents, python code, and reasoning.
You are allowed to produce ONLY steps with the following JSON schemas:
{allowed_steps}
If you want to create a python code step, output a python code block inside backticks, like this:
```python
a = 1
result = a*2
print(result)
```
Note that the last code line should print the result.
Do not reproduce the schema when producing steps; use it as a reference.
"""

FACTS_SURVEY = f"""Before we begin executing the plan, please answer the following pre-survey to the best of your ability.
Keep in mind that you are Ken Jennings-level with trivia, and Mensa-level with puzzles, so there should be a deep well to draw from.
Here is the pre-survey:

    1. Please list any specific facts or figures that are GIVEN in the request itself. It is possible that there are none.
    2. Please list any facts that may need to be looked up, and WHERE SPECIFICALLY they might be found. In some cases, authoritative sources are mentioned in the request itself.
    3. Please list any facts that may need to be derived (e.g., via logical deduction, simulation, or computation)
    4. Please list any facts that are recalled from memory, hunches, well-reasoned guesses, etc.

When answering this survey, keep in mind that "facts" will typically be specific names, dates, statistics, etc. You should produce following sections:
    1. Given facts
    2. Facts to look up
    3. Facts to derive
    4. Educated guesses

Respond with thought kind="facts_survey_thought"! Follow the provided JSON schema and do not replicate the schema fields itself!
{FORMAT}
"""

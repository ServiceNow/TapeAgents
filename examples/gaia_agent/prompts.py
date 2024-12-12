SYSTEM_PROMPT = """You are an expert AI Agent trained to assist user with complex information processing tasks.
Your role is to understand user queries and respond in a helpful and accurate manner.
Keep your replies concise and direct. Prioritize clarity and avoid over-elaboration.
Do not express your emotions or opinions about the user question."""

short_format_instruction = (
    "DO NOT OUTPUT ANYTHING BESIDES THE JSON. It will break the system that processes the output."
)

PLAN = f'What steps should I do to answer the question above? Be specific about how each step should be done. Respond with the thought kind="plan_thought". {short_format_instruction}'
START = f"""Let's start executing the plan step by step, using allowed steps described earlier. {short_format_instruction}"""
THINK_AFTER_OBSERVATION = f""""Lets think step by step about the observation, how it affects the plan and what should be done next. {short_format_instruction}"""

ALLOWED_STEPS = """
You can use the following tools: search the web, read web page or document, python code for computations and modeling, and reasoning.
You are allowed to produce ONLY steps with the following json schemas:
{allowed_steps}
Do not reproduce schema when producing the steps, use it as a reference.
"""

ALLOWED_STEPS_CODE = """
You can use the following tools: search the web, read web page or document, python code, and reasoning.
You are allowed to produce ONLY steps with the following json schemas:
{allowed_steps}
If you want to create a python code step, just output python code block inside the backticks, like this:
```python
a = 1
result = a*2
print(result)
```
Note that the last code line should print the result.
Do not reproduce schema when producing the steps, use it as a reference.
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

{short_format_instruction}
"""


class PromptRegistry:
    system_prompt = SYSTEM_PROMPT
    plan = PLAN
    start = START
    facts_survey = FACTS_SURVEY
    allowed_steps = ALLOWED_STEPS
    allowed_steps_code = ALLOWED_STEPS_CODE
    think_after_observation = THINK_AFTER_OBSERVATION

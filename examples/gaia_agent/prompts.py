SYSTEM_PROMPT = """You are an expert AI Agent trained to assist users with complex information processing tasks.
Your role is to understand user queries and respond in a helpful and accurate manner.
Keep your replies concise and direct. Prioritize clarity and avoid over-elaboration.
Do not express emotions or opinions about user questions."""

FORMAT = "Output only a single step. DO NOT OUTPUT ANYTHING BESIDES THE JSON! DO NOT PLACE ANY COMMENTS INSIDE THE JSON. It will break the system that processes the output."

PLAN = f'What steps should I take to answer this question? Be specific about how each step should be performed. Respond with the thought kind="plan_thought". {FORMAT}'

START = f"""Let's start executing the plan step by step, using the allowed steps described earlier. {FORMAT}"""

REFLECT_AND_ACT = """
Produce a JSON list with two steps in it.
The first step be the reasoning thought that was produced according to the following rules:
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
The second step should be a action that performs the proposed step.
"""

VERIFY = f"""
First, state the answer that was found.
Then observe and summarize the entire history of interaction as a list.
Then consider how you can verify the answer.
Finally, propose a list of steps for verification.
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

SYSTEM_PROMPT = """You are an expert AI Agent trained to assist user with complex information processing tasks.
Your role is to understand user queries and respond in a helpful and accurate manner.
Keep your replies concise and direct. Prioritize clarity and avoid over-elaboration.
Do not express your emotions or opinions about the user question."""

short_format_instruction = (
    "DO NOT OUTPUT ANYTHING BESIDES THE JSON. It will break the system that processes the output."
)

PLAN = f'What steps should I do to answer the question above? Be specific about how each step should be done. Respond with the thought kind="plan_thought". {short_format_instruction}'
PLAN3 = f'What steps should I do to answer the question above? Propose 3 different plans to follow. Be specific about how each step should be done. Respond with the thought kind="draft_plans_thought". {short_format_instruction}'
BETTER_PLAN = f'Now, considering the draft plans, facts, sources and availabe steps, propose a new balanced and more detailed plan to follow. Respond with the thought kind="plan_thought". {short_format_instruction}'
SOURCES_PLAN = f"For the facts that we need to lookup, create the list of sources where to find them. Respond with the sources_thought. {short_format_instruction}"
START_EXECUTION = f"""Let's start executing the plan step by step, using allowed steps described earlier. {short_format_instruction}"""
THINK_AFTER_OBSERVATION = f"""Produce the reasoning thought step with the thoughts about recent observation and the overall state of the task. {short_format_instruction}"""
THINK_AFTER_CALCULATION = f"""Produce the reasoning thought step with the thoughts about plausbility and sensibility of the results of the recent calculation. {short_format_instruction}"""

ALLOWED_STEPS = """
You can use the following tools: search the web, read web page or document, extract fact from the web page or document, python code, and reasoning.
You are allowed to produce ONLY steps with the following json schemas:
{allowed_steps}
Do not reproduce schema when producing the steps, use it as a reference.
"""

FILENAME = "The question is about {ext} file {filename}"

MLM = """Write a detailed caption for this image. Pay special attention to any details that might be useful for someone answering the following:

{prompt}
"""

FACTS_SURVEY = f"""Before we begin executing the plan, please answer the following pre-survey to the best of your ability. 
Keep in mind that you are Ken Jennings-level with trivia, and Mensa-level with puzzles, so there should be a deep well to draw from.
For each fact provide the description, expected json-compatible format and, if possible, measurement unit. 
The fact name should be short and in lowercase. The description should be detailed, self-sustained and informative.
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

Respond with list_of_facts_thought.
{short_format_instruction}
"""

IS_SUBTASK_FINISHED = """Assess if the subtask objective has been fully achieved. If the objective has been achieved or if we're stuck, finish working on the subtask by producing finish_subtask_thought with the status and description.
If the objective has not been achieved, produce the next step.
"""


class PromptRegistry:
    system_prompt = SYSTEM_PROMPT
    plan = PLAN
    plan3 = PLAN3
    better_plan = BETTER_PLAN
    sources_plan = SOURCES_PLAN
    start_execution = START_EXECUTION
    filename = FILENAME
    mlm = MLM
    facts_survey = FACTS_SURVEY
    allowed_steps = ALLOWED_STEPS
    is_subtask_finished = IS_SUBTASK_FINISHED
    think_after_observation = THINK_AFTER_OBSERVATION
    think_after_calculation = THINK_AFTER_CALCULATION

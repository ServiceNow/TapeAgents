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
You can use the following tools: search the web, read web page or document, extract fact from the web page or document, calculator, and reasoning.
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

FACTS_SURVEY_V2_SYSTEM = """Below I will present you a request. Before we begin addressing the request, please answer the following pre-survey to the best of your ability. Keep in mind that you are Ken Jennings-level with trivia, and Mensa-level with puzzles, so there should be a deep well to draw from.

Here is the request:"""

FACTS_SURVEY_V2_GUIDANCE = """
Here is the pre-survey:

    1. Please list any specific facts or figures that are GIVEN in the request itself. It is possible that there are none.
    2. Please list any facts that may need to be looked up, and WHERE SPECIFICALLY they might be found. In some cases, authoritative sources are mentioned in the request itself.
    3. Please list any facts that may need to be derived (e.g., via logical deduction, simulation, or computation)
    4. Please list any facts that are recalled from memory, hunches, well-reasoned guesses, etc.

When answering this survey, keep in mind that "facts" will typically be specific names, dates, statistics, etc. Your answer should use headings:

    1. GIVEN OR VERIFIED FACTS
    2. FACTS TO LOOK UP
    3. FACTS TO DERIVE
    4. EDUCATED GUESSES

DO NOT include any other headings or sections in your response. DO NOT list next steps or plans until asked to do so.
"""

ALLOWED_STEPS_V2 = """
Produce answer with the following json schema:
{schema}
Do not reproduce schema when producing the step, use it only as a reference!
DO NOT OUTPUT ANYTHING BESIDES THE JSON. It will break the system that processes the output.
"""

TEAM = """
WebSurfer: A helpful assistant with access to a web browser. Ask them to perform web searches, open pages, and interact with content (e.g., clicking links, scrolling the viewport, etc., filling in form fields, etc.) It can also summarize the entire page, or answer questions based on the content of the page. It can also be asked to sleep and wait for pages to load, in cases where the pages seem to be taking a while to load.
Coder: A helpful and general-purpose AI programmer that has strong language skills, Python skills, and Linux command line skills. Avoid using for reading common file formats, as FileSurfer is more proficient at this. Can be helpful to process previously extracted numerical facts.
FileSurfer: An agent that can handle reading of local files only, could be helpful to read documents attached to the task or downloaded from the web by WebSurfer. Proficient with PDF, DOCX, XLSX, CSV, PPTX and other common formats.
"""

PLAN_V2 = """To address this request we have following tools:

Web search, web surfing, python code execution, reading local files, reasoning.

Based on the available tools and known and unknown facts, please reason step by step and produce shor draft of the plan to solve this task.
After initial reasoning and drafting, produce more detailed bullet-point plan for addressing the original request. For each step of the plan, provide the following:
- detailed description of things to do
- list of tools and expected outcomes like concrete artifacts at the end of the step: facts, files, documents, or data to be produced
- prerequisites, a list of the results of the previous steps, or known facts needed to start working on this step.
"""

FORMALIZE_SYSTEM_PROMPT = """
You are an expert AI Agent trained to produce complex json structure from the plain text input.
"""

FORMALIZE_INPUT = """
Plain text input to be converted into json structure:
{content}
"""

FORMALIZE_GUIDANCE = """
Please produce the json structure from the plain text input in the previous message.
"""

FORMALIZE_FORMAT = """
Produce step using the following json schema:
{schema}
Do not reproduce schema fields when producing the step, use it only as a reference!
DO NOT OUTPUT ANYTHING BESIDES THE JSON. It will break the system that processes the output.
"""

START_EXECUTION_V2 = """
Let's start executing given task, using allowed steps described earlier.
Briefly describe required steps.
"""

TODO_NEXT = """
Let's think what to do next.
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

    facts_survey_v2_system = FACTS_SURVEY_V2_SYSTEM
    facts_survey_v2 = FACTS_SURVEY_V2_GUIDANCE
    allowed_steps_v2 = ALLOWED_STEPS_V2
    plan_v2 = PLAN_V2
    formalize_system_prompt = FORMALIZE_SYSTEM_PROMPT
    formalize_guidance = FORMALIZE_GUIDANCE
    formalize_input = FORMALIZE_INPUT
    formalize_format = FORMALIZE_FORMAT
    start_execution_v2 = START_EXECUTION_V2
    todo_next = TODO_NEXT

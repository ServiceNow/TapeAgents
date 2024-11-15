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

Based on the available tools and known and unknown facts, please reason step by step and produce short draft of the plan to solve this task.
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
Let's start executing given task. Briefly describe required steps. After that think what to do next.
"""

TODO_NEXT = """
Describe single immediate next step to be done.
"""

REFLECT_PLAN_STEP_RESULT = """
Reflect on the results of executing the subtask solving the plan step:
- If the subtask was successfully completed, compare the result with the expected outcome. If the expected outcome was not achieved, reflect on the reasons for the discrepancy.
- If the subtask was failed, reflect on the reasons for the failure. Think out loud how should we change the plan to be able to complete the task anyway.
"""

REFLECT_PLAN_STATUS = """
Reflect on the current state of the plan execution:
- If all the steps are completed, reflect on the overall success of the plan.
- If some steps are failed, reflect on the reasons for the failure. Think out loud how should we change the plan to be able to complete the task anyway.

"""

PLAN_STATUS = """
Current plan status:
Total steps: {total_steps}
Completed steps: {completed_steps}
Remaining steps: {remaining_steps}
"""

REPLAN = """Our previous attempt to solve task failed.
Our previous plan:
{plan}
Description of the failure:
{failure}


Please reason step by step and produce short draft of the new plan to solve this task. It should be different from the previous one.
Produce detailed bullet-point plan for addressing the original request. For each step of the plan, provide the following:
- detailed description of things to do
- list of tools and expected outcomes like concrete artifacts at the end of the step: facts, files, documents, or data to be produced
- prerequisites, a list of the results of the previous steps, or known facts needed to start working on this step.
"""

FINAL_ANSWER = """
Read the above messages and output a FINAL ANSWER to the question. The question is repeated here for convenience:

{task}

To output the final answer, use the following template: FINAL ANSWER: [YOUR FINAL ANSWER]
Your FINAL ANSWER should be a number OR as few words as possible OR a comma separated list of numbers and/or strings.
ADDITIONALLY, your FINAL ANSWER MUST adhere to any formatting instructions specified in the original question (e.g., alphabetization, sequencing, units, rounding, decimal places, etc.)
If you are asked for a number, express it numerically (i.e., with digits rather than words), don't use commas, and DO NOT INCLUDE UNITS such as $ or USD or percent signs unless specified otherwise.
If you are asked for a string, don't use articles or abbreviations (e.g. for cities), unless specified otherwise. Don't output any final sentence punctuation such as '.', '!', or '?'.
If you are asked for a comma separated list, apply the above rules depending on whether the elements are numbers or strings.
If you are unable to determine the final answer, output empty result.
"""

REFLECT_OBSERVATION = """
Reflect on the results of the recent observation.
Check if the expected outcome was achieved.
Check if the observation was sufficient or not.
Check if the observation was inconclusive.
If the was some error, reflect on the reasons for the error and its content.
"""

FACTS_SURVEY_UPDATE = """
As a reminder, we are working to solve the following task:

{task}

It's clear we aren't making as much progress as we would like, but we may have learned something new. Please rewrite the following fact sheet, updating it to include anything new we have learned that may be helpful. Example edits can include (but are not limited to) adding new guesses, moving educated guesses to verified facts if appropriate, etc. Updates may be made to any section of the fact sheet, and more than one section of the fact sheet can be edited. This is an especially good time to update educated guesses, so please at least add or update one educated guess or hunch, and explain your reasoning.

Here is the old fact sheet:

1. GIVEN OR VERIFIED FACTS
    {given}
2. FACTS TO LOOK UP
    {lookup}
3. FACTS TO DERIVE
    {derive}
4. EDUCATED GUESSES
    {guesses}

Respond with updated facts sheet.
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
    reflect_plan_step_result = REFLECT_PLAN_STEP_RESULT
    reflect_plan_status = REFLECT_PLAN_STATUS
    plan_status = PLAN_STATUS
    replan = REPLAN
    final_answer = FINAL_ANSWER
    reflect_observation = REFLECT_OBSERVATION
    facts_survey_update = FACTS_SURVEY_UPDATE

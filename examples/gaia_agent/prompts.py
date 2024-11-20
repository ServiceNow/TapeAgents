SYSTEM_PROMPT = """You are an expert AI Agent trained to help user by solving complex information-processing tasks.
Your role is to solve the task using the best of your abilities and knowledge.
Keep your replies concise and direct. Prioritize clarity and avoid over-elaboration.
Do not express your emotions or opinions about the task.
"""

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
You can use the following tools: search the web, read web page or document, extract fact from the web page or document, execute python code, and reasoning.
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

FACTS_SURVEY_V2_SYSTEM = """
You are an expert AI Agent trained to assist user with complex information processing tasks.
Keep in mind that you are Ken Jennings-level with trivia, and Mensa-level with puzzles, so there should be a deep well to draw from.
"""

FACTS_SURVEY_V2_GUIDANCE = """
Please answer the following pre-survey to the best of your ability:
    1. List any specific facts or figures that are GIVEN in the request itself. It is possible that there are none.
    2. List any facts that may need to be looked up, and WHERE SPECIFICALLY they might be found. In some cases, authoritative sources are mentioned in the request itself.
    3. List any facts that may need to be derived (e.g., via logical deduction, simulation, or computation)
    4. List any facts that are recalled from memory, hunches, well-reasoned guesses, etc.

When answering this survey, keep in mind that "facts" will typically be specific names, dates, statistics, etc. Your answer should use headings:

    1. GIVEN FACTS
    2. FACTS TO LOOK UP
    3. FACTS TO DERIVE
    4. EDUCATED GUESSES

Sometimes, the request will not contain any given facts, facts to look up, or facts to derive. In such cases, you may leave those sections blank.
DO NOT include any other headings or sections in your response. DO NOT list next steps or plans until asked to do so.
"""

tools = """You can use the following tools:
- Wikipedia search
- Web search
- Web browsing
- Reading local documents
- Extracting fact from the web page or document
- Python code execution
- Reasoning"""

ALLOWED_STEPS_V2 = """
You are allowed to produce ONLY steps described in this json schema:
{schema}
Do not reproduce schema when producing the step, use it only as a reference!
DO NOT OUTPUT ANYTHING BESIDES THE JSON. It will break the system that processes the output.
"""

TEAM = """
WebSurfer: A helpful assistant with access to a web browser. Ask them to perform web searches, open pages, and interact with content (e.g., clicking links, scrolling the viewport, etc., filling in form fields, etc.) It can also summarize the entire page, or answer questions based on the content of the page. It can also be asked to sleep and wait for pages to load, in cases where the pages seem to be taking a while to load.
Coder: A helpful and general-purpose AI programmer that has strong language skills, Python skills, and Linux command line skills. Avoid using for reading common file formats, as FileSurfer is more proficient at this. Can be helpful to process previously extracted numerical facts.
FileSurfer: An agent that can handle reading of local files only, could be helpful to read documents attached to the task or downloaded from the web by WebSurfer. Proficient with PDF, DOCX, XLSX, CSV, PPTX and other common formats.
"""

PLAN_V2 = f"""What steps should I do to answer the question above? Be specific about how each step should be done.

{tools}

For each step in the plan, include:
- A detailed description of the tasks to perform.
- A list of required tools.
- A list of expected outcomes, such as facts, files, documents, or data.
- A list of prerequisites, including any results from previous steps or known facts necessary to proceed. First step should not have prerequisites.
"""

START_EXECUTION_V2 = """
Briefly describe how to solve this task.
"""

REFLECT_SUBTASK = """
Evaluate the execution of the subtask in the plan step:
- If the subtask was successfully completed, describe the achieved result.
- If the subtask failed, analyze the reasons for the failure and suggest adjustments to the plan to enable successful task completion.
"""

REFLECT_PLAN_STATUS = """
Assess the current state of plan execution:

Determine whether the produced results contain all the necessary information to answer the question.
If there is enough information to answer the question, mark task as solved and plan as finished.
If plan is not finished solved, evaluate the progress of plan completion.
If any steps have failed, analyze the reasons for the failure and suggest adjustments to the plan to ensure the task can still be completed.
"""

PLAN_STATUS = """
Current plan status:
Total steps: {total_steps}
Completed steps: {completed_steps}
Remaining steps: {remaining_steps}

Facts found:
{facts}
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
Provide reasoning about the current state of the task after the last observation.
"""

FACTS_SURVEY_UPDATE = """
As a reminder, we are working to solve the following task:

{task}

We're executing the following plan:

{plan}

Here is the current fact sheet:
{facts}

We've completed another step in the plan, here is the results:

{last_results}

Please update the following fact sheet to include any new information we have learned that may be helpful. Update rules:
- Edit should update found facts if new facts have been reported in results.
- Edit can include adding new guesses, moving educated guesses to found facts where applicable, etc.
- You can update any section of the fact sheet, with multiple sections edited if necessary.
- Remove educated guesses that have been proven false or replaced by found facts.
- Do not remove previously found facts!

Respond with updated facts sheet.
"""

REASON = "Let's think step by step."
ACT = "Work only on the current subtask! Produce result step when the current subtask is solved."


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
    start_execution_v2 = START_EXECUTION_V2
    reflect_subtask = REFLECT_SUBTASK
    reflect_plan_status = REFLECT_PLAN_STATUS
    plan_status = PLAN_STATUS
    replan = REPLAN
    final_answer = FINAL_ANSWER
    reflect_observation = REFLECT_OBSERVATION
    facts_survey_update = FACTS_SURVEY_UPDATE
    reason = REASON
    act = ACT

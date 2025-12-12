"""
Generic Agent prompts from AgentLab.
Based on: https://github.com/ServiceNow/AgentLab/blob/main/src/agentlab/agents/generic_agent/generic_agent_prompt.py
"""

# System prompt for generic agent
GENERIC_SYSTEM_PROMPT = """\
You are an agent trying to solve a web task based on the content of the page and
user instructions. You can interact with the page and explore, and send messages to the user. Each time you
submit an action it will be sent to the browser and you will receive a new page."""

# Goal instructions template
GENERIC_GOAL_INSTRUCTIONS = """\
# Instructions
Review the current state of the page and all other information to find the best
possible next action to accomplish your goal. Your answer will be interpreted
and executed by a program, make sure to follow the formatting instructions.

## Goal:
{goal}"""

# Hints for the agent
GENERIC_HINTS = """\
Note:
* Some tasks may be game like and may require to interact with the mouse position
in x, y coordinates.
* Some text field might have auto completion. To see it, you have to type a few
characters and wait until next step.
* If you have to cut and paste, don't forget to select the text first.
* Coordinate inside an SVG are relative to it's top left corner.
* Make sure to use bid to identify elements when using commands.
* Interacting with combobox, dropdowns and auto-complete fields can be tricky,
sometimes you need to use select_option, while other times you need to fill
or click and wait for the reaction of the page.
* Only one action is allowed per step. Do not provide multiple actions.
"""

# Be cautious instruction
GENERIC_BE_CAUTIOUS = """\
Be very cautious. Avoid submitting anything before verifying the effect of your
actions. Take the time to explore the effect of safe actions first. For example
you can fill a few elements of a form, but don't click submit before verifying
that everything was filled correctly.
"""

# Thinking prompt
GENERIC_THINK_PROMPT = ""
GENERIC_THINK_ABSTRACT = """\
<think>Think step by step. If you need to make calculations such as coordinates, write them here. Describe the effect
that your previous action had on the current content of the page.</think>"""

GENERIC_THINK_CONCRETE = """\
<think>From previous action I tried to set the value of year to "2022",
using select_option, but it doesn't appear to be in the form. It may be a
dynamic dropdown, I will try using click with the bid "a324" and look at the
response from the page.</think>"""

# Plan prompt
GENERIC_PLAN_PROMPT = """\
# Plan:
You just executed step {plan_step} of the previously proposed plan:
{previous_plan}
After reviewing the effect of your previous actions, verify if your plan is still
relevant and update it if necessary."""

GENERIC_PLAN_ABSTRACT = """\
<plan>Provide a multi step plan that will guide you to accomplish the goal. There
should always be steps to verify if the previous action had an effect. The plan
can be revisited at each steps. Specifically, if there was something unexpected.
The plan should be cautious and favor exploring befor submitting.</plan>
<step>Integer specifying the step of current action</step>"""

GENERIC_PLAN_CONCRETE = """\
<plan>
1. fill form (failed)
   * type first name
   * type last name
2. Try to activate the form
   * click on tab 2
3. fill form again
   * type first name
   * type last name
4. verify and submit
   * verify form is filled
   * submit if filled, if not, replan
</plan>
<step>2</step>"""

# Criticise prompt
GENERIC_CRITICISE_PROMPT = ""
GENERIC_CRITICISE_ABSTRACT = """\
<action_draft>Write a first version of what you think is the right action.</action_draft>
<criticise>Criticise action_draft. What could be wrong with it? Enumerate reasons why it
could fail. Did your past actions had the expected effect? Make sure you're not
repeating the same mistakes.</criticise>"""

GENERIC_CRITICISE_CONCRETE = """\
<action_draft>click("32")</action_draft>
<criticise>click("32") might not work because the element is not visible yet. I need to
explore the page to find a way to activate the form.</criticise>"""

# Memory prompt
GENERIC_MEMORY_PROMPT = ""
GENERIC_MEMORY_ABSTRACT = """\
<memory>Write down anything you need to remember in the future.</memory>"""

GENERIC_MEMORY_CONCRETE = """\
<memory>The form requires activation before it can be filled. Tab 2 seems to contain
the activation button.</memory>"""

# Action prompt info
GENERIC_ACTION_SET_INFO = """\
Note: This action set allows you to interact with your environment. Most of them
are python function executing playwright code. The primary way of referring to
elements in the page is through bid which are specified in your observations."""

# Complete action space description (from BrowserGym)
GENERIC_ACTION_SPACE_DESCRIPTION = """\
10 different types of actions are available.

fill(bid: str, value: str, enable_autocomplete_menu: bool = False)
    Examples:
        fill('45', 'multi-line\\nexample')

        fill('a12', 'example with "quotes"')

        fill('b534', 'Montre', True)

select_option(bid: str, options: str | list[str])
    Examples:
        select_option('a48', 'blue')

        select_option('c48', ['red', 'green', 'blue'])

click(bid: str, button: Literal['left', 'middle', 'right'] = 'left', modifiers: list[typing.Literal['Alt', 'Control', 'ControlOrMeta', 'Meta', 'Shift']] = [])
    Examples:
        click('a51')

        click('b22', button='right')

        click('48', button='middle', modifiers=['Shift'])

hover(bid: str)
    Examples:
        hover('b8')

press(bid: str, key_comb: str)
    Examples:
        press('88', 'Backspace')

        press('a26', 'ControlOrMeta+a')

        press('a61', 'Meta+Shift+t')

click_coordinates(x: float, y: float, button: Literal['left', 'middle', 'right'] = 'left')
    Examples:
        click_coordinates(120, 340)

        click_coordinates(512.5, 280.25, button='right')

go_back()
    Examples:
        go_back()

go_forward()
    Examples:
        go_forward()

send_msg_to_user(text: str)
    Examples:
        send_msg_to_user('Based on the results of my search, the city was built in 1751.')

Only a single action can be provided at once. Example:
fill('b534', 'Montre', True)
"""

# Complete prompt structure template
GENERIC_COMPLETE_PROMPT_TEMPLATE = """\
# Instructions
Review the current state of the page and all other information to find the best
possible next action to accomplish your goal. Your answer will be interpreted
and executed by a program, make sure to follow the formatting instructions.

## Goal:
{goal}

# Observation of current step:

{observation}

{history}

# Action space:
{action_space}

{hints}

# Abstract Example

Here is an abstract version of the answer with description of the content of
each tag. Make sure you follow this structure, but replace the content with your
answer:

<think>
Think step by step. If you need to make calculations such as coordinates, write them here. Describe the effect
that your previous action had on the current content of the page.
</think>

<action>
One single action to be executed. You can only use one action at a time.
</action>


# Concrete Example

Here is a concrete example of how to format your answer.
Make sure to follow the template with proper tags:

<think>
From previous action I tried to set the value of year to "2022",
using select_option, but it doesn't appear to be in the form. It may be a
dynamic dropdown, I will try using click with the bid "a324" and look at the
response from the page.
</think>

<action>
click('a324')
</action>
"""

# Allowed steps/actions prompt (to be filled with actual schemas)
GENERIC_ALLOWED_STEPS = """\
# Action space:
{action_set_info}

You are allowed to produce ONLY steps with the following json schemas:
{allowed_steps}
Do not reproduce schema when producing the steps, use it as a reference.
"""

# Abstract example for action
GENERIC_ACTION_ABSTRACT = """\
<action>{example_action}</action>"""

GENERIC_ACTION_CONCRETE = """\
<action>click('a324')</action>"""

# Abstract example structure
GENERIC_ABSTRACT_EXAMPLE = """\
# Abstract Example

Here is an abstract version of the answer with description of the content of
each tag. Make sure you follow this structure, but replace the content with your
answer:
{think_abstract}
{plan_abstract}
{memory_abstract}
{criticise_abstract}
{action_abstract}
"""

# Concrete example structure
GENERIC_CONCRETE_EXAMPLE = """\
# Concrete Example

Here is a concrete example of how to format your answer.
Make sure to follow the template with proper tags:
{think_concrete}
{plan_concrete}
{memory_concrete}
{criticise_concrete}
{action_concrete}
"""

# History prompt template
GENERIC_HISTORY_HEADER = "# History of interaction with the task:\n"
GENERIC_HISTORY_STEP = "## step {step_number}\n\n{action_view}\n"
GENERIC_HISTORY_ERROR = "Error from previous action: {error}\n"

# MAC hint
GENERIC_MAC_HINT = "\nNote: you are on mac so you should use Meta instead of Control for Control+C etc.\n"

# BID info
GENERIC_BID_INFO = """\
Note: [bid] is the unique alpha-numeric identifier at the beginning of lines for
each element in the AXTree. Always use bid to refer to elements in your actions."""

# Coordinate notes
GENERIC_COORD_CENTER_NOTE = """\
Note: center coordinates are provided in parenthesis and are relative to the top
left corner of the page."""

GENERIC_COORD_BOX_NOTE = """\
Note: bounding box of each object are provided in parenthesis and are relative
to the top left corner of the page."""

# Visible elements note
GENERIC_VISIBLE_ELEMENTS_NOTE = """\
Note: only elements that are visible in the viewport are presented. You might
need to scroll the page, or open tabs or menus to see more."""

# Visible tag note
GENERIC_VISIBLE_TAG_NOTE = """\
Note: You can only interact with visible elements. If the "visible" tag is not
present, the element is not visible on the page."""


class GenericPromptRegistry:
    """Registry for generic agent prompts."""
    system_prompt = GENERIC_SYSTEM_PROMPT
    goal_instructions = GENERIC_GOAL_INSTRUCTIONS
    hints = GENERIC_HINTS
    be_cautious = GENERIC_BE_CAUTIOUS
    
    think_prompt = GENERIC_THINK_PROMPT
    think_abstract = GENERIC_THINK_ABSTRACT
    think_concrete = GENERIC_THINK_CONCRETE
    
    plan_prompt = GENERIC_PLAN_PROMPT
    plan_abstract = GENERIC_PLAN_ABSTRACT
    plan_concrete = GENERIC_PLAN_CONCRETE
    
    criticise_prompt = GENERIC_CRITICISE_PROMPT
    criticise_abstract = GENERIC_CRITICISE_ABSTRACT
    criticise_concrete = GENERIC_CRITICISE_CONCRETE
    
    memory_prompt = GENERIC_MEMORY_PROMPT
    memory_abstract = GENERIC_MEMORY_ABSTRACT
    memory_concrete = GENERIC_MEMORY_CONCRETE
    
    action_set_info = GENERIC_ACTION_SET_INFO
    action_space_description = GENERIC_ACTION_SPACE_DESCRIPTION
    allowed_steps = GENERIC_ALLOWED_STEPS
    action_abstract = GENERIC_ACTION_ABSTRACT
    action_concrete = GENERIC_ACTION_CONCRETE
    
    abstract_example = GENERIC_ABSTRACT_EXAMPLE
    concrete_example = GENERIC_CONCRETE_EXAMPLE
    complete_prompt_template = GENERIC_COMPLETE_PROMPT_TEMPLATE
    
    history_header = GENERIC_HISTORY_HEADER
    history_step = GENERIC_HISTORY_STEP
    history_error = GENERIC_HISTORY_ERROR
    
    mac_hint = GENERIC_MAC_HINT
    bid_info = GENERIC_BID_INFO
    coord_center_note = GENERIC_COORD_CENTER_NOTE
    coord_box_note = GENERIC_COORD_BOX_NOTE
    visible_elements_note = GENERIC_VISIBLE_ELEMENTS_NOTE
    visible_tag_note = GENERIC_VISIBLE_TAG_NOTE

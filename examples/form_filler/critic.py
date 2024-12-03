from typing import Any, Generator, Literal, TypeAlias, Union
from tapeagents.core import Action, Prompt, Tape, Thought, SetNextNode
from tapeagents.dialog_tape import AssistantStep, UserStep
from tapeagents.llms import LLM, LLMStream
from tapeagents.agent import Agent, Node
from tapeagents.view import Call, Respond

from llmd2.tapeagents_tmp.ghreat.schema import FunctionSchema
from llmd2.tapeagents_tmp.ghreat.steps import I_NOTE_STEPS, I_SHOULD_STEPS, FunctionCandidates, RequestFunctionParameters
from llmd2.tapeagents_tmp.ghreat.tape import FormFillerTape, from_llmd2_dialog_dict


class IsGrounded(Thought):
    kind: Literal["is_grounded"] = "is_grounded"
    reasoning: str
    grounded: bool

class ContradictsHistory(Thought):
    kind: Literal["contradicts_history"] = "contradicts_history"
    reasoning: str
    contradicts_history: bool

class ContradictsGroundingStatement(Thought):
    kind: Literal["contradicts_grounding_statement"] = "contradicts_grounding_statement"
    reasoning: str
    contradicts_grounding_statement: bool

class FollowsAgentThoughts(Thought):
    kind: Literal["follows_agent_thoughts"] = "follows_agent_thoughts"
    reasoning: str
    follows_agent_thoughts: bool

class FollowsConversationHistory(Thought):
    kind: Literal["follows_conversation_history"] = "follows_conversation_history"
    reasoning: str
    follows_conversation_history: bool

class FollowsGroundingStatement(Thought):
    kind: Literal["follows_grounding_statement"] = "follows_grounding_statement"
    reasoning: str
    follows_grounding_statement: bool



class IsHelpful(Thought):
    kind: Literal["is_helpful"] = "is_helpful"
    reasoning: str
    helpful: bool

class AgentAsksForIntent(Thought):
    kind: Literal["agent_asks_for_intent"] = "agent_asks_for_intent"
    reasoning: str
    agent_asks_for_intent: bool

class AgentAsksForSlotValues(Thought):
    kind: Literal["agent_asks_for_slot_values"] = "agent_asks_for_slot_values"
    reasoning: str
    agent_asks_for_slot_values: bool

class AgentAsksForConfirmation(Thought):
    kind: Literal["agent_asks_for_confirmation"] = "agent_asks_for_confirmation"
    reasoning: str
    agent_asks_for_confirmation: bool


class IsResponsive(Thought):
    kind: Literal["is_responsive"] = "is_responsive"
    reasoning: str
    responsive: bool

class UserAsksQuestion(Thought):
    kind: Literal["user_asks_question"] = "user_asks_question"
    reasoning: str
    user_asks_question: bool

class AgentAnswersQuestion(Thought):
    kind: Literal["agent_answers_question"] = "agent_answers_question"
    reasoning: str
    agent_answers_question: bool

class ValidInfoResponsive(Thought):
    kind: Literal["valid_info_responsive"] = "valid_info_responsive"
    reasoning: str
    valid_info_responsive: bool

class InvalidInfoResponsive(Thought):
    kind: Literal["invalid_info_responsive"] = "invalid_info_responsive"
    reasoning: str
    invalid_info_responsive: bool


class IsAccurate(Thought):
    kind: Literal["is_accurate"] = "is_accurate"
    reasoning: str
    accurate: bool

class UserProvidesInfo(Thought):
    kind: Literal["user_provides_info"] = "user_provides_info"
    reasoning: str
    provides_info: bool



class IsTransparent1(Thought):
    kind: Literal["is_transparent_i_should"] = "is_transparent_i_should"
    reasoning: str
    transparent1: bool



class IsTransparent2(Thought):
    kind: Literal["is_transparent_i_note"] = "is_transparent_i_note"
    reasoning: str
    transparent2: bool


class Critique(Action):
    kind: Literal["critique"] = "critique"
    grounded: bool
    helpful: bool
    responsive: bool
    accurate: bool
    transparent1: bool
    transparent2: bool


CriticStep = Union[
    IsGrounded,
    FollowsAgentThoughts,
    FollowsConversationHistory,
    FollowsGroundingStatement,
    ContradictsHistory,
    ContradictsGroundingStatement,
    IsHelpful,
    AgentAsksForIntent,
    AgentAsksForSlotValues,
    AgentAsksForConfirmation,
    IsResponsive,
    UserAsksQuestion,
    AgentAnswersQuestion,
    ValidInfoResponsive,
    InvalidInfoResponsive,
    IsAccurate,
    UserProvidesInfo,
    IsTransparent1,
    IsTransparent2,
    Critique,
    Call,
    Respond,
    SetNextNode
]
CriticTape: TypeAlias = Tape[FormFillerTape, CriticStep]
Critic: TypeAlias = Agent[CriticTape]


class CriticExpert(Critic):
    @classmethod
    def create(cls, llm: LLM, templates: dict[str,dict[str, str]]):
        subagents = [
            # GroundednessExpert
            Critic.create(
                llm,
                templates = templates["is_grounded_templates"],
                name="groundedness_expert",
                nodes=[ContradictsHistoryNode(), ContradictsGroundingStatementNode(), GroundednessNode()]
            ),
            # HelpfulnessExpert
            Critic.create(
                llm,
                templates=templates["is_helpful_templates"],
                name="helpfulness_expert",
                nodes=[HelpfulnessRoutingNode(), AskedForIntentNode(), SlotValuesRequestedNode(), ConfirmationRequestedNode(), AgentAnswersUserNode(), HelpfulnessAggregationNode()]
            ),  # do some jumps between nodes based on the intent discovered or not
            # ResponsivenessExpert
            Critic.create(
                llm,
                templates=templates["is_responsive_templates"],
                name="responsiveness_expert",
                nodes=[UserAsksQuestionNode(), AgentAnswersQuestionNode(), ValidInfoResponsiveNode(), InvalidInfoResponsiveNode(), ResponsivenessNode()]
            ),
            # AccuracyExpert
            Critic.create(
                llm,
                templates=templates["is_accurate_templates"],
                name="accuracy_expert",
                nodes=[UserProvidesInfoNode(), AccuracyNode()]
            ),
            # TransparencyExpert1
            Critic.create(
                llm,
                templates=templates["is_transparent1_templates"],
                name="transparency1_expert",
                nodes=[TransparencyNode(name="transparency1_node", mode="i_should")]
            ),
            # TransparencyExpert2
            Critic.create(
                llm,
                templates=templates["is_transparent2_templates"],
                name="transparency2_expert",
                nodes=[TransparencyNode(name="transparency2_node", mode="i_note")]
            ),
        ]
        nodes = [
            CallNextAgentNode(name="call_groundedness_node", agent_name="groundedness_expert"),
            CallNextAgentNode(name="call_helpfulness_node", agent_name="helpfulness_expert"),
            CallNextAgentNode(name="call_responsiveness_node", agent_name="responsiveness_expert"),
            CallNextAgentNode(name="call_accuracy_node", agent_name="accuracy_expert"),
            CallNextAgentNode(name="call_transparency1_node", agent_name="transparency1_expert"),
            CallNextAgentNode(name="call_transparency2_node", agent_name="transparency2_expert"),
            CriticNode()
        ]
        return super().create(name="critique", subagents=subagents, nodes=nodes)


class CallNextAgentNode(Node):
    agent_name: str

    def generate_steps(self, agent: Critic, tape: CriticTape, llm_stream: LLMStream):
        yield Call(agent_name=self.agent_name)


class CriticNode(Node):
    name: str = "critic_node"

    def generate_steps(self, agent: CriticExpert, tape: CriticTape, llm_stream: LLMStream):
        # check if we have resutls from all our subagents
        grounded = None
        helpful = None
        responsive = None
        accurate = None
        transparent1 = None
        transparent2 = None
        for step in tape.steps:
            if isinstance(step, IsGrounded):
                grounded = step.grounded
            elif isinstance(step, IsHelpful):
                helpful = step.helpful
            elif isinstance(step, IsResponsive):
                responsive = step.responsive
            elif isinstance(step, IsAccurate):
                accurate = step.accurate
            elif isinstance(step, IsTransparent1):
                transparent1 = step.transparent1
            elif isinstance(step, IsTransparent2):
                transparent2 = step.transparent2
        if grounded is None or helpful is None or accurate is None or responsive is None or transparent1 is None or transparent2 is None:
            raise ValueError(f"Not all subagents have provided results. Tape: {tape.steps}")
        yield Critique(
            grounded=grounded,
            helpful=helpful,
            responsive=responsive,
            accurate=accurate,
            transparent1=transparent1,
            transparent2=transparent2
        )


def create_template_values(tape: FormFillerTape) -> dict:
    template_values = {
        "candidate_functions": [],
        "current_function_schema": None,
        "current_function_description": None,
        "predicted_function_name": None,
        "dialogue_history": [],
        "dialogue_history_with_notes": [],
        "i_note_steps_history": [],
        "last_question": "",
        "last_user_message": "",
        "last_agent_message": "",
        "i_should_steps": [],
        "i_note_steps": [],
        "requested_parameter": "Agent thought (hidden from User): I should not ask for any parameters.",
        "grounding_statement_3": ""
    }

    context_steps, predicted_steps = tape.get_context_and_predicted_steps()
    context_steps = list(context_steps)  # convert to list to be able to access the last element
    # assert that the last steps are AssistantStep - UserStep
    assert isinstance(context_steps[-2], AssistantStep), f"Second last step is not an AssistantStep: {context_steps[-2]}"
    assert isinstance(context_steps[-1], UserStep), f"Last step is not a UserStep: {context_steps[-1]}"
    template_values["last_question"] = f"{context_steps[-2].content}"
    template_values["last_user_message"] = f"{context_steps[-1].content}"
    
    current_function_schema = None

    for step in context_steps:
        if isinstance(step, FunctionCandidates):
            template_values["candidate_functions"].extend(step.candidates)
        elif isinstance(step, UserStep):
            template_values["dialogue_history"].append(f"User: {step.content}")
            template_values["dialogue_history_with_notes"].append(f"User: {step.content}")
        elif isinstance(step, AssistantStep):
            template_values["dialogue_history"].append(f"Agent: {step.content}")
            template_values["dialogue_history_with_notes"].append(f"Agent: {step.content}")
        elif isinstance(step, FunctionSchema):
            current_function_schema = step
            template_values["current_function_schema"] = step.model_dump_json(indent=2, exclude={"return_value"})
            template_values["current_function_description"] = step.description
        elif isinstance(step, I_NOTE_STEPS):
            template_values["i_note_steps_history"].append(f"Agent note (hidden from User): {step.llm_view()}")
            template_values["dialogue_history_with_notes"].append(f"Agent note (hidden from User): {step.llm_view()}")
    template_values["dialogue_history"] = "\n".join(template_values["dialogue_history"])
    template_values["dialogue_history_with_notes"] = "\n".join(template_values["dialogue_history_with_notes"])

    requested_parameter_name = ""
    for step in predicted_steps:
        if isinstance(step, FunctionCandidates):
            # after the first use message, the resolve_function_step is in the predicted steps
            template_values["candidate_functions"].extend(step.candidates)
        elif isinstance(step, FunctionSchema):
            current_function_schema = step
            template_values["current_function_schema"] = step.model_dump_json(indent=2, exclude={"return_value"})
            template_values["current_function_description"] = step.description
            template_values["predicted_function_name"] = step.name
        elif isinstance(step, AssistantStep):
            template_values["last_agent_message"] = step.content
        elif isinstance(step, I_SHOULD_STEPS):
            template_values["i_should_steps"].append(f"Agent thought (hidden from User): {step.llm_view()}")
            if isinstance(step, RequestFunctionParameters):
                requested_parameter_name = step.parameters[0]
        elif isinstance(step, I_NOTE_STEPS):
            template_values["i_note_steps"].append(f"Agent note (hidden from User): {step.llm_view()}")
            template_values["i_note_steps_history"].append(f"Agent note (hidden from User): {step.llm_view()}")
    template_values["i_should_steps"] = "\n".join(template_values["i_should_steps"])
    template_values["i_note_steps"] = "\n".join(template_values["i_note_steps"])
    template_values["i_note_steps_history"] = "\n".join(template_values["i_note_steps_history"])
    template_values["candidate_functions"] = f"[\n  " + ",\n".join([c.model_dump_json(indent=2) for c in template_values["candidate_functions"]]) + "\n]"

    # get all the details about the requested parameter
    if current_function_schema and requested_parameter_name:
        default_value = current_function_schema.get_parameter_default(requested_parameter_name)
        default = f" (default: {default_value})" if default_value else ""
        optional = current_function_schema.is_optional_parameter(requested_parameter_name)
        enum_values = current_function_schema.get_parameter_enum_values(requested_parameter_name)
        enum = f" (enum values: {enum_values})" if enum_values else ""
        template_values["requested_parameter"] = f"Agent thought (hidden from User): I should ask for {requested_parameter_name} (optional: {optional}){default}{enum}."

    # get the grounding statement #3: "the agent can only do the current request"
    if template_values["current_function_schema"]:
        filled_parameters = tape.get_filled_parameters_as_llm_view()
        template_values["grounding_statement_3"] = f"The Agent can only help the user complete the current request:\n{template_values['current_function_schema']}\n{filled_parameters}"
    else:
        template_values["grounding_statement_3"] = f"The Agent can only help the user with the following requests: {template_values['candidate_functions']}"

    return template_values


class HelpfulnessRoutingNode(Node):
    name: str = "helpfulness_routing_node"

    def generate_steps(self, agent: Critic, tape: CriticTape, llm_stream: LLMStream):
        assert tape.context, "Context is required to generate steps"
        if tape.context.intent_is_discovered:
            # if intent is discovered, jump to the SlotValuesRequestedNode (index #2)
            yield SetNextNode(next_node=2)
        else:
            # otherwise, jump to the AskedForIntentNode (index #1)
            yield SetNextNode(next_node=1)


class AskedForIntentNode(Node):
    name: str = "asked_for_intent_node"

    def make_prompt(self, agent: Critic, tape: CriticTape) -> Prompt:
        assert tape.context, "Context is required to build a prompt"
        template_variables = create_template_values(tape.context)
        messages = [
            {"role": "system", "content": agent.templates["asks_for_intent_system_prompt"]},
            {"role": "user", "content": agent.templates["asks_for_intent_user_prompt"].format(**template_variables)},
        ]
        return Prompt(messages=messages)

    def generate_steps(self, agent: Critic, tape: CriticTape, llm_stream: LLMStream):
        text = llm_stream.get_text()
        yield AgentAsksForIntent(reasoning=f"completion={text}", agent_asks_for_intent=text.lower().strip('.') == "yes")
        # this is all we need to check helpfulness, jump to last node (index #5)
        yield SetNextNode(next_node=5)


class SlotValuesRequestedNode(Node):
    name: str = "slot_values_requested_node"

    def make_prompt(self, agent: Critic, tape: CriticTape) -> Prompt:
        assert tape.context, "Context is required to build a prompt"
        template_variables = create_template_values(tape.context)
        messages = [
            {"role": "system", "content": agent.templates["slot_values_requested_system_prompt"]},
            {"role": "user", "content": agent.templates["slot_values_requested_user_prompt"].format(**template_variables)},
        ]
        return Prompt(messages=messages)
    
    def generate_steps(self, agent: Critic, tape: CriticTape, llm_stream: LLMStream):
        text = llm_stream.get_text()
        yield AgentAsksForSlotValues(reasoning=f"completion={text}", agent_asks_for_slot_values=text.lower().strip('.') == "yes")


class ConfirmationRequestedNode(Node):
    name: str = "confirmation_requested_node"

    def make_prompt(self, agent: Critic, tape: CriticTape) -> Prompt:
        assert tape.context, "Context is required to build a prompt"
        template_variables = create_template_values(tape.context)
        messages = [
            {"role": "system", "content": agent.templates["confirmation_requested_system_prompt"]},
            {"role": "user", "content": agent.templates["confirmation_requested_user_prompt"].format(**template_variables)},
        ]
        return Prompt(messages=messages)
    
    def generate_steps(self, agent: Critic, tape: CriticTape, llm_stream: LLMStream):
        text = llm_stream.get_text()
        yield AgentAsksForConfirmation(reasoning=f"completion={text}", agent_asks_for_confirmation=text.lower().strip('.') == "yes")


class AgentAnswersUserNode(Node):
    name: str = "agent_answers_user_node"

    def make_prompt(self, agent: Critic, tape: CriticTape) -> Prompt:
        assert tape.context, "Context is required to build a prompt"
        template_variables = create_template_values(tape.context)
        messages = [
            {"role": "system", "content": agent.templates["question_answered_system_prompt"]},
            {"role": "user", "content": agent.templates["question_answered_user_prompt"].format(**template_variables)},
        ]
        return Prompt(messages=messages)
    
    def generate_steps(self, agent: Critic, tape: CriticTape, llm_stream: LLMStream):
        text = llm_stream.get_text()
        yield AgentAnswersQuestion(reasoning=f"completion={text}", agent_answers_question=text.lower().strip('.') == "yes")


class HelpfulnessAggregationNode(Node):
    name: str = "helpfulness_node"

    def generate_steps(self, agent: Critic, tape: CriticTape, llm_stream: LLMStream):
        assert tape.context, "Context is required to generate steps"
        if tape.context.intent_is_discovered:
            # if intent is discovered, grab results from 3 nodes
            slot_values_requested = None
            confirmation_requested = None
            agent_answers_user = None
            for step in tape.steps:
                if isinstance(step, AgentAsksForSlotValues):
                    slot_values_requested = step.agent_asks_for_slot_values
                elif isinstance(step, AgentAsksForConfirmation):
                    confirmation_requested = step.agent_asks_for_confirmation
                elif isinstance(step, AgentAnswersQuestion):
                    agent_answers_user = step.agent_answers_question

            if slot_values_requested is None or confirmation_requested is None or agent_answers_user is None:
                raise ValueError(f"Not all subagents have provided results. Tape: {tape.steps}")
            yield IsHelpful(
                reasoning=f"slot_values_requested={slot_values_requested}. confirmation_requested={confirmation_requested}. agent_answers_user={agent_answers_user}",
                helpful=(slot_values_requested or confirmation_requested) and agent_answers_user
            )
        else:
            # if intent is not discovered, grab the result from the intent discovery node
            agent_asks_for_intent = None
            for step in tape.steps:
                if isinstance(step, AgentAsksForIntent):
                    agent_asks_for_intent = step.agent_asks_for_intent
            if agent_asks_for_intent is None:
                raise ValueError(f"Not all subagents have provided results. Tape: {tape.steps}")
            yield IsHelpful(reasoning=f"agent_asks_for_intent={agent_asks_for_intent}", helpful=agent_asks_for_intent)
        yield Respond()


class TransparencyNode(Node):
    mode: Literal["i_should", "i_note"]

    def make_prompt(self, agent: Critic, tape: CriticTape) -> Prompt:
        assert tape.context, "Context is required to build a prompt"

        # no need to build a prompt when there is nothing to check
        if self.mode == "i_should" and not tape.context.predicted_i_should:
            return Prompt()
        if self.mode == "i_note" and not tape.context.predicted_i_note:
            return Prompt()

        template_variables = create_template_values(tape.context)
        messages = [
            {"role": "system", "content": agent.templates["system_prompt"]},
            {"role": "user", "content": agent.templates["user_prompt"].format(**template_variables)},
        ]
        return Prompt(messages=messages)

    def generate_steps(self, agent: Critic, tape: CriticTape, llm_stream: LLMStream) -> Generator[CriticStep, None, None]:
        assert tape.context, "Context is required to generate steps"
        if self.mode == "i_should":
            if tape.context.predicted_i_should:
                text = llm_stream.get_text()
                yield IsTransparent1(reasoning=f"completion={text}", transparent1=text.lower().strip('.') == "yes")
            else:
                yield IsTransparent1(reasoning="I don't see any thoughts in the predicted steps", transparent1=True)
        elif self.mode == "i_note":
            if tape.context.predicted_i_note:
                text = llm_stream.get_text()
                yield IsTransparent2(reasoning=f"completion={text}", transparent2=text.lower().strip('.') == "yes")
            else:
                yield IsTransparent2(reasoning="I don't see any notes in the predicted steps", transparent2=True)
        yield Respond()


class GroundednessNode(Node):
    name: str = "groundedness_node"

    def generate_steps(self, agent: Critic, tape: CriticTape, llm_stream: LLMStream):
        # check if we have resutls from all our subagents
        contradicts_history = None
        contradicts_grounding_statement = None
        for step in tape.steps:
            if isinstance(step, ContradictsHistory):
                contradicts_history = step.contradicts_history
            elif isinstance(step, ContradictsGroundingStatement):
                contradicts_grounding_statement = step.contradicts_grounding_statement

        if contradicts_history is None or contradicts_grounding_statement is None:
            raise ValueError(f"Not all subagents have provided results. Tape: {tape.steps}")
        yield IsGrounded(
            reasoning=f"contradicts_history={contradicts_history}. \
                contradicts_grounding_statement={contradicts_grounding_statement}",
            grounded=not(contradicts_history or contradicts_grounding_statement)
        )
        yield Respond()


class ContradictsHistoryNode(Node):
    name: str = "contradicts_history_node"

    def make_prompt(self, agent: Critic, tape: CriticTape) -> Prompt:
        template_variables = create_template_values(tape.context)
        messages = [
            {"role": "system", "content": agent.templates["contradicts_history_system_prompt"]},
            {"role": "user", "content": agent.templates["contradicts_history_user_prompt"].format(**template_variables)},
        ]
        return Prompt(messages=messages)

    def generate_steps(self, agent: Critic, tape: CriticTape, llm_stream: LLMStream):
        text = llm_stream.get_text()
        yield ContradictsHistory(reasoning=f"completion={text}", contradicts_history=text.lower().strip('.') == "yes")
        # yield Respond()


class ContradictsGroundingStatementNode(Node):
    name: str = "contradicts_grounding_statement_node"

    def make_prompt(self, agent: Critic, tape: CriticTape) -> Prompt:
        template_variables = create_template_values(tape.context)
        messages = [
            {"role": "system", "content": agent.templates["contradicts_statement_system_prompt"]},
            {"role": "user", "content": agent.templates["contradicts_statement_user_prompt"].format(**template_variables)},
        ]
        return Prompt(messages=messages)

    def generate_steps(self, agent: Critic, tape: CriticTape, llm_stream: LLMStream):
        text = llm_stream.get_text()
        yield ContradictsGroundingStatement(reasoning=f"completion={text}", contradicts_grounding_statement=text.lower().strip('.') == "yes")
        # yield Respond()


class FollowsConversationHistoryNode(Node):
    name: str = "follows_conversation_history_node"

    def make_prompt(self, agent: Critic, tape: CriticTape) -> Prompt:
        template_variables = create_template_values(tape.context)
        messages = [
            {"role": "system", "content": agent.templates["system_prompt"]},
            {"role": "user", "content": agent.templates["user_prompt"].format(**template_variables)},
        ]
        return Prompt(messages=messages)

    def generate_steps(self, agent: Critic, tape: CriticTape, llm_stream: LLMStream):
        text = llm_stream.get_text()
        yield FollowsConversationHistory(reasoning=f"completion={text}", follows_conversation_history=text.lower().strip('.') == "yes")
        yield Respond()


class FollowsAgentThoughtsNode(Node):
    name: str = "follows_agent_thoughts_node"

    def make_prompt(self, agent: Critic, tape: CriticTape) -> Prompt:
        template_variables = create_template_values(tape.context)
        messages = [
            {"role": "system", "content": agent.templates["system_prompt"]},
            {"role": "user", "content": agent.templates["user_prompt"].format(**template_variables)},
        ]
        return Prompt(messages=messages)

    def generate_steps(self, agent: Critic, tape: CriticTape, llm_stream: LLMStream):
        i_note_present = tape.context.i_note_in_context or tape.context.predicted_i_note
        if i_note_present:
            text = llm_stream.get_text()
            yield FollowsAgentThoughts(reasoning=f"completion={text}", follows_agent_thoughts=text.lower().strip('.') == "yes")
        else:
            yield FollowsAgentThoughts(reasoning="I don't see any notes in the conversation", follows_agent_thoughts=True)
        yield Respond()


class FollowsGroundingStatementNode(Node):
    name: str = "follows_grounding_statement_node"

    def make_prompt(self, agent: Critic, tape: CriticTape) -> Prompt:
        template_variables = create_template_values(tape.context)
        messages = [
            {"role": "system", "content": agent.templates["system_prompt"]},
            {"role": "user", "content": agent.templates["user_prompt"].format(**template_variables)},
        ]
        return Prompt(messages=messages)
    
    def generate_steps(self, agent: Critic, tape: CriticTape, llm_stream: LLMStream):
        text = llm_stream.get_text()
        yield FollowsGroundingStatement(reasoning=f"completion={text}", follows_grounding_statement=text.lower().strip('.') == "yes")
        yield Respond()


class AccuracyNode(Node):
    name: str = "accuracy_node"

    def make_prompt(self, agent: Critic, tape: CriticTape) -> Prompt:
        assert tape.context, "Context is required to build a prompt"
        template_variables = create_template_values(tape.context)
        # check if agent has to do intent discovery
        if tape.context.intent_is_discovered:
            template_to_use = "correct_notes"
        else:
            template_to_use = "correct_intent"
        messages = [
            {"role": "system", "content": agent.templates[f"{template_to_use}_system_prompt"]},
            {"role": "user", "content": agent.templates[f"{template_to_use}_user_prompt"].format(**template_variables)},
        ]
        return Prompt(messages=messages)

    def generate_steps(self, agent: Critic, tape: CriticTape, llm_stream: LLMStream):
        # check if we have restuls from our subagent
        user_provides_info = None
        for step in tape.steps:
            if isinstance(step, UserProvidesInfo):
                user_provides_info = step.provides_info
                break
        if user_provides_info is None:
            raise ValueError(f"Not all subagents have provided results. Tape: {tape.steps}")

        assert tape.context, "Context is required to generate steps"
        # check if agent already did intent discovery
        if tape.context.intent_is_discovered:
            if user_provides_info:
                if tape.context.predicted_i_note:
                    text = llm_stream.get_text()
                    yield IsAccurate(reasoning=f"completion={text}", accurate=text== "Yes")
                else:
                    yield IsAccurate(
                        reasoning="The user provided some values, but I don't see any notes in the predicted steps.",
                        accurate=False
                    )
            else:
                # user did not provide any values
                if tape.context.predicted_i_note:
                    yield IsAccurate(
                        reasoning="The user did not provide any values, but I see notes in the predicted steps.",
                        accurate=False
                    )
                else:
                    yield IsAccurate(
                        reasoning="The user did not provide any values, and I don't see any notes in the predicted steps.",
                        accurate=True
                    )
        # agent is doing intent discovery
        else:
            if user_provides_info:
                text = llm_stream.get_text()
                yield IsAccurate(reasoning=f"completion={text}", accurate=text.lower().strip('.') == "yes")
            else:
                yield IsAccurate(
                    reasoning="The user did not give a valid intent, nothing to check.",
                    accurate=True
                )
        yield Respond()


class UserProvidesInfoNode(Node):
    name: str = "user_provides_info_node"

    def make_prompt(self, agent: Critic, tape: CriticTape) -> Prompt:
        assert tape.context, "Context is required to build a prompt"
        # check if agent has to do intent discovery
        if tape.context.intent_is_discovered:
            template_to_use = "info"
        else:
            template_to_use = "intent"

        template_variables = create_template_values(tape.context)
        messages = [
            {"role": "system", "content": agent.templates[f"user_provides_{template_to_use}_system_prompt"]},
            {"role": "user", "content": agent.templates[f"user_provides_{template_to_use}_user_prompt"].format(**template_variables)},
        ]
        return Prompt(messages=messages)

    def generate_steps(self, agent: Critic, tape: CriticTape, llm_stream: LLMStream):
        text = llm_stream.get_text()
        yield UserProvidesInfo(reasoning=f"completion={text}", provides_info=text.lower().strip('.') == "yes")


class ResponsivenessNode(Node):

    def generate_steps(self, agent: Critic, tape: CriticTape, llm_stream: LLMStream):
        # check if we have resutls from all our subagents
        user_asks_question = None
        agent_answers_question = None
        valid_info_responsive = None
        invalid_info_responsive = None

        for step in tape.steps:
            if isinstance(step, UserAsksQuestion):
                user_asks_question = step.user_asks_question
            elif isinstance(step, AgentAnswersQuestion):
                agent_answers_question = step.agent_answers_question
            elif isinstance(step, ValidInfoResponsive):
                valid_info_responsive = step.valid_info_responsive
            elif isinstance(step, InvalidInfoResponsive):
                invalid_info_responsive = step.invalid_info_responsive

        if user_asks_question is None or agent_answers_question is None or valid_info_responsive is None or invalid_info_responsive is None:
            raise ValueError(f"Not all subagents have provided results. Tape: {tape.steps}")

        yield IsResponsive(
            reasoning=f"user_asks_question={user_asks_question}. \
                agent_answers_question={agent_answers_question}. \
                valid_info_responsive={valid_info_responsive}. \
                invalid_info_responsive={invalid_info_responsive}",
            responsive=agent_answers_question and valid_info_responsive and invalid_info_responsive
        )
        yield Respond()


class UserAsksQuestionNode(Node):
    name: str = "user_asks_question_node"

    def make_prompt(self, agent: Critic, tape: CriticTape) -> Prompt:
        assert tape.context, "Context is required to build a prompt"
        template_variables = create_template_values(tape.context)
        messages = [
            {"role": "system", "content": agent.templates["user_asks_question_system_prompt"]},
            {"role": "user", "content": agent.templates["user_asks_question_user_prompt"].format(**template_variables)},
        ]
        return Prompt(messages=messages)
    
    def generate_steps(self, agent: Critic, tape: CriticTape, llm_stream: LLMStream):
        text = llm_stream.get_text()
        yield UserAsksQuestion(reasoning=f"completion={text}", user_asks_question=text.lower().strip('.') == "yes")


class AgentAnswersQuestionNode(Node):
    name: str = "agent_answers_question_node"

    def make_prompt(self, agent: Critic, tape: CriticTape) -> Prompt:
        assert tape.context, "Context is required to build a prompt"
        user_asks_question = None
        for step in tape.steps:
            if isinstance(step, UserAsksQuestion):
                user_asks_question = step.user_asks_question
                break
        if user_asks_question is None:
            raise ValueError(f"UserAsksQuestion step is missing. Tape: {tape.steps}")

        if user_asks_question:
            template_variables = create_template_values(tape.context)
            messages = [
                {"role": "system", "content": agent.templates["agent_answers_question_system_prompt"]},
                {"role": "user", "content": agent.templates["agent_answers_question_user_prompt"].format(**template_variables)},
            ]
            return Prompt(messages=messages)
        else:
            # no question was asked, no need to check for an answer
            return Prompt()
    
    def generate_steps(self, agent: Critic, tape: CriticTape, llm_stream: LLMStream):
        user_asks_question = None
        for step in tape.steps:
            if isinstance(step, UserAsksQuestion):
                user_asks_question = step.user_asks_question
                break
        if user_asks_question is None:
            raise ValueError(f"UserAsksQuestion step is missing. Tape: {tape.steps}")
        if user_asks_question:
            text = llm_stream.get_text()
            yield AgentAnswersQuestion(reasoning=f"completion={text}", agent_answers_question=text.lower().strip('.') == "yes")
        else:
            yield AgentAnswersQuestion(reasoning="user did not ask a question.", agent_answers_question=True)


class ValidInfoResponsiveNode(Node):
    name: str = "valid_info_responsive_node"

    def make_prompt(self, agent: Critic, tape: CriticTape) -> Prompt:
        assert tape.context, "Context is required to build a prompt"
        template_variables = create_template_values(tape.context)
        messages = [
            {"role": "system", "content": agent.templates["valid_slot_values_system_prompt"]},
            {"role": "user", "content": agent.templates["valid_slot_values_user_prompt"].format(**template_variables)},
        ]
        return Prompt(messages=messages)
    
    def generate_steps(self, agent: Critic, tape: CriticTape, llm_stream: LLMStream):
        text = llm_stream.get_text()
        yield ValidInfoResponsive(reasoning=f"completion={text}", valid_info_responsive=text.lower().strip('.') == "yes")


class InvalidInfoResponsiveNode(Node):
    name: str = "invalid_info_responsive_node"

    def make_prompt(self, agent: Critic, tape: CriticTape) -> Prompt:
        assert tape.context, "Context is required to build a prompt"
        template_variables = create_template_values(tape.context)
        messages = [
            {"role": "system", "content": agent.templates["invalid_slot_values_system_prompt"]},
            {"role": "user", "content": agent.templates["invalid_slot_values_user_prompt"].format(**template_variables)},
        ]
        return Prompt(messages=messages)
    
    def generate_steps(self, agent: Critic, tape: CriticTape, llm_stream: LLMStream):
        text = llm_stream.get_text()
        yield InvalidInfoResponsive(reasoning=f"completion={text}", invalid_info_responsive=text.lower().strip('.') == "yes")


def from_llmd2_annotated_dialog_dict(d: dict) -> CriticTape:
    form_filler_tape = from_llmd2_dialog_dict(d)
    # get golden label is it exists
    annotation = [ann for ann in d.get("annotation", []) if ann["gold"]]
    if not annotation:
        annotation = d.get("annotation", [])
    assert len(annotation) == 1, d.get("annotation", [])
    annotation = annotation[0]
    # reconstruct critique steps from annotations
    critique_steps: list[CriticStep] = []
    for label in annotation.get("labels", []):
        if label == "grounded_choices":
            critique_steps.append(IsGrounded(reasoning="", grounded=annotation["labels"][label] == "Yes"))
        elif label == "helpful_choices":
            critique_steps.append(IsHelpful(reasoning="", helpful=annotation["labels"][label] == "Yes"))
        elif label == "responsive_choices":
            critique_steps.append(IsResponsive(reasoning="", responsive=annotation["labels"][label] == "Yes"))
        elif label == "accurate_choices":
            critique_steps.append(IsAccurate(reasoning="", accurate=annotation["labels"][label] == "Yes"))
        elif label == "transparent_choices1":
            critique_steps.append(IsTransparent1(reasoning="", transparent1=annotation["labels"][label] == "Yes"))
        elif label == "transparent_choices2":
            critique_steps.append(IsTransparent2(reasoning="", transparent2=annotation["labels"][label] == "Yes"))
        else:
            raise ValueError(f"Unexpected label: {label}")
    # critique_steps.append(Critique(
    #     grounded=annotation.get("labels", []).get("grounded_choices", "No") == "Yes",
    #     helpful=annotation.get("labels", []).get("helpful_choices", "No") == "Yes",
    #     responsive=annotation.get("labels", []).get("responsive_choices", "No") == "Yes",
    #     accurate=annotation.get("labels", []).get("accurate_choices", "No") == "Yes",
    #     transparent1=annotation.get("labels", []).get("transparent_choices1", "No") == "Yes",
    #     transparent2=annotation.get("labels", []).get("transparent_choices2", "No") == "Yes",
    # ))
    return CriticTape(context=form_filler_tape, steps=critique_steps)

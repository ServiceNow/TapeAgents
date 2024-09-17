import logging

from tapeagents.environment import Environment
from tapeagents.tools import BasicToolbox
from tapeagents.utils import FatalError

from .steps import (
    ActionExecutionFailure,
    AgentResponseParsingFailureAction,
    CalculationResultObservation,
    CodeResultObservation,
    ConvertFactAction,
    GaiaAction,
    NextPageAction,
    PageObservation,
    PythonCodeAction,
    ReadDocumentAction,
    SearchAction,
    SearchResultsObservation,
    UseCalculatorAction,
)
from .tape import GaiaTape

logger = logging.getLogger(__name__)


class GaiaEnvironment(Environment):
    def __init__(self, tools: BasicToolbox) -> None:
        super().__init__()
        self.tools = tools

    def react(self, tape: GaiaTape) -> GaiaTape:
        actions = [step for step in tape.steps[-tape.metadata.n_added_steps :] if isinstance(step, GaiaAction)]
        for action in actions:
            try:
                match action:
                    case SearchAction():
                        tape = tape.append(self.search(action))
                    case ReadDocumentAction():
                        tape = tape.append(self.read_document(action))
                    case NextPageAction():
                        tape = tape.append(self.next_page(action))
                    case ConvertFactAction():
                        tape = tape.append(self.convert(action))
                    case UseCalculatorAction():
                        tape = tape.append(self.calculate(action))
                    case PythonCodeAction():
                        tape = tape.append(self.run_python_code(action))
                    case AgentResponseParsingFailureAction():
                        pass
                    case _:
                        raise Exception(f"Unknown action: {type(action)}")
            except FatalError:
                raise
            except Exception as e:
                logger.exception(f"Error during action execution: {e}")
                tape = tape.append(ActionExecutionFailure(error=str(e)))
                break
        return tape

    def convert(self, action: ConvertFactAction) -> CalculationResultObservation:
        result = self.tools.calculate(
            action.expression, {"value": action.fact_value, action.original_fact_name: action.fact_value}
        )
        return CalculationResultObservation(name=action.converted_fact_name, result=result)

    def calculate(self, action: UseCalculatorAction) -> CalculationResultObservation:
        result = self.tools.calculate(action.expression, action.facts or {})
        return CalculationResultObservation(name=action.fact_name, result=result)

    def run_python_code(self, action: PythonCodeAction) -> CodeResultObservation:
        result, stdout, stderr = self.tools.run_python_code(action.code, action.facts or {})
        return CodeResultObservation(name=action.fact_name, result=result, stdout=stdout, stderr=stderr)

    def read_document(self, action: ReadDocumentAction) -> PageObservation:
        text, current_page, total_pages = self.tools.get_page(action.url)
        return PageObservation(text=text, current_page=current_page, total_pages=total_pages)

    def next_page(self, action: NextPageAction) -> PageObservation:
        text, current_page, total_pages = self.tools.get_next_page()
        return PageObservation(text=text, current_page=current_page, total_pages=total_pages)

    def search(self, action) -> SearchResultsObservation:
        if "web" not in action.source and "wiki" not in action.source:
            raise ValueError(f"Supported sources are 'web' and 'wiki', got {action.source}")
        last_serp = self.tools.websearch(action.query, source=action.source)
        return SearchResultsObservation(query=action.query, serp=last_serp)

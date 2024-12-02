import logging
from pathlib import Path

from tapeagents.core import Action
from tapeagents.environment import Environment
from tapeagents.steps import WatchVideoAction
from tapeagents.tools.calculator import calculate
from tapeagents.tools.media_reader import get_video_observation
from tapeagents.tools.python_interpreter import python_calculate, run_python_code
from tapeagents.tools.simple_browser import SimpleTextBrowser
from tapeagents.utils import FatalError

from .steps import (
    ActionExecutionFailure,
    CalculationResultObservation,
    CodeResultObservation,
    ConvertFactAction,
    LLMOutputParsingFailureAction,
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
    def __init__(self, safe_calculator: bool = True, attachment_dir: str = "attachments", **kwargs) -> None:
        super().__init__()
        self.attachment_dir = attachment_dir
        self.browser = SimpleTextBrowser(**kwargs)
        self.calculate = calculate if safe_calculator else python_calculate

    def react(self, tape: GaiaTape) -> GaiaTape:
        actions = [step for step in tape.steps[-tape.metadata.n_added_steps :] if isinstance(step, Action)]
        for action in actions:
            try:
                match action:
                    case SearchAction():
                        if action.source == "wiki":
                            query = f"site:wikipedia.org {action.query}"
                        elif action.source == "youtube":
                            query = f"site:youtube.com {action.query}"
                        elif action.source == "web":
                            query = action.query
                        else:
                            raise ValueError(f"Supported sources are 'web', 'wiki' and 'youtube', got {action.source}")

                        try:
                            serp = self.browser.get_search_results(query)
                        except Exception as e:
                            raise FatalError(f"Failed to get search results: {e}")
                        tape = tape.append(SearchResultsObservation(query=action.query, serp=serp))
                    case ReadDocumentAction():
                        text, total_pages, error = self.browser.get_page(action.url)
                        tape = tape.append(
                            PageObservation(
                                text=text,
                                current_page=1,
                                total_pages=total_pages,
                                error=error or None,
                            )
                        )
                    case NextPageAction():
                        text, current_page, total_pages = self.browser.get_next_page()
                        tape = tape.append(
                            PageObservation(
                                text=text,
                                current_page=current_page,
                                total_pages=total_pages,
                                error=self.browser._page_error if self.browser._page_error else None,
                            )
                        )
                    case WatchVideoAction():
                        video_observation = get_video_observation(
                            action.video_url, self.attachment_dir, action.start_time, action.end_time
                        )
                        tape = tape.append(video_observation)
                    case ConvertFactAction():
                        result = self.calculate(
                            action.expression,
                            {"value": action.fact_value, action.original_fact_name: action.fact_value},
                        )
                        tape = tape.append(CalculationResultObservation(name=action.converted_fact_name, result=result))
                    case UseCalculatorAction():
                        result = self.calculate(action.expression, action.facts or {})
                        tape = tape.append(CalculationResultObservation(name=action.fact_name, result=result))
                    case PythonCodeAction():
                        result, stdout, stderr = run_python_code(action.code, action.facts or {})
                        tape = tape.append(
                            CodeResultObservation(name=action.fact_name, result=result, stdout=stdout, stderr=stderr)
                        )
                    case LLMOutputParsingFailureAction():
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

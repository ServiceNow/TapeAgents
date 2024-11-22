import logging
from pathlib import Path

from tapeagents.core import Action
from tapeagents.environment import Environment
from tapeagents.steps import GetVideoAction, VideoObservation
from tapeagents.tools.calculator import calculate
from tapeagents.tools.media_reader import watch_video
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
    def __init__(self, safe_calculator: bool = True, attachment_dir: str = "attachment", **kwargs) -> None:
        super().__init__()
        self.attachment_dir = attachment_dir
        self.browser = SimpleTextBrowser(**kwargs)
        self.calculate = calculate if safe_calculator else python_calculate

        # Create attachment directory if it doesn't exist
        Path(attachment_dir).mkdir(parents=True, exist_ok=True)

    def react(self, tape: GaiaTape) -> GaiaTape:
        actions = [step for step in tape.steps[-tape.metadata.n_added_steps :] if isinstance(step, Action)]
        for action in actions:
            try:
                match action:
                    case SearchAction():
                        if "web" not in action.source and "wiki" not in action.source:
                            raise ValueError(f"Supported sources are 'web' and 'wiki', got {action.source}")
                        query = f"site:wikipedia.org {action.query}" if "wiki" in action.source else action.query
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
                    case GetVideoAction():
                        video_path, video_contact_sheet_paths, thumbnail_path, subtitle_path, subtitle_text, error = (
                            watch_video(action.video_url, action.start_time, self.attachment_dir)
                        )
                        tape = tape.append(
                            VideoObservation(
                                video_path=video_path,
                                video_contact_sheet_paths=video_contact_sheet_paths,
                                thumbnail_path=thumbnail_path,
                                subtitle_path=subtitle_path,
                                subtitle_text=subtitle_text,
                                error=error or None,
                            )
                        )
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

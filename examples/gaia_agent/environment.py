import logging
import os
import shutil

from tapeagents.container_executor import CodeBlock, ContainerExecutor
from tapeagents.environment import Environment
from tapeagents.tools.calculator import calculate
from tapeagents.tools.python_interpreter import run_python_code
from tapeagents.tools.simple_browser import SimpleTextBrowser
from tapeagents.utils import FatalError

from .steps import (
    ActionExecutionFailure,
    CalculationResultObservation,
    CodeResultObservation,
    ConvertFactAction,
    GaiaAction,
    GaiaQuestion,
    GaiaStep,
    ImageObservation,
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
    def __init__(
        self,
        code_sandbox: ContainerExecutor | None = None,
        image_observations: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        self.code_sandbox = code_sandbox
        self.image_observations = image_observations
        self.browser = SimpleTextBrowser(**kwargs)

    def react(self, tape: GaiaTape) -> GaiaTape:
        actions = [step for step in tape.steps[-tape.metadata.n_added_steps :] if isinstance(step, GaiaAction)]
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
                    case ConvertFactAction():
                        result = calculate(
                            action.expression,
                            {"value": action.fact_value, action.original_fact_name: action.fact_value},
                        )
                        tape = tape.append(CalculationResultObservation(name=action.converted_fact_name, result=result))
                    case UseCalculatorAction():
                        result = calculate(action.expression, action.facts or {})
                        tape = tape.append(CalculationResultObservation(name=action.fact_name, result=result))
                    case PythonCodeAction():
                        if self.code_sandbox is not None:
                            result = self.code_sandbox.execute_code_blocks(
                                [CodeBlock(code=print_last_line(action.code), language="python")]
                            )
                            obs = CodeResultObservation(
                                name=action.fact_name,
                                result=result.output.strip(),
                                stdout=f"Exit code: {result.exit_code}",
                                stderr="",
                            )
                        else:
                            # TODO: remove this option and permutations crutch
                            logger.warning(f"Code sandbox is not provided, running code locally!\n{action.code}")
                            if "permutations" in action.code:
                                result, stdout, stderr = "", "", "Execution timeout"
                            else:
                                result, stdout, stderr = run_python_code(action.code, {})
                            obs = CodeResultObservation(
                                name=action.fact_name,
                                result=result,
                                stdout=stdout,
                                stderr=stderr,
                            )
                        tape = tape.append(obs)
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

    def task_to_observations(
        self,
        task: dict,
        max_doc_length: int = 8000,
    ) -> list[GaiaStep]:
        question = GaiaQuestion.from_task(task)
        image = None
        if question.filename:
            name, ext = question.filename.rsplit(".", maxsplit=1)
            ext = ext.lower()
            if ext == "zip":
                folder_name = name
                os.makedirs(folder_name, exist_ok=True)
                shutil.unpack_archive(question.filename, folder_name)
                document_text = "\n\nArchive contains the following files:\n"
                for i, file in enumerate(os.listdir(folder_name)):
                    file_path = os.path.join(folder_name, file)
                    content = self.browser.get_whole_document(file_path)
                    file_text = f"{i+1}. {file}. Content:\n{content}\n\n"
                    if len(file_text) > max_doc_length:
                        file_text = ""
                    file_text += f"{i+1}. Path to the '{file}': {file_path}"
                    document_text += file_text
            elif ext in ("png", "jpg", "jpeg") and self.image_observations:
                image = ImageObservation(
                    image_path=question.filename,
                    caption="Attached image",
                )
                document_text = ""
            else:
                content = self.browser.get_whole_document(question.filename)
                document_text = f"\n\n{ext.upper()} document content:\n{content}\n"
                if len(document_text) > max_doc_length:
                    document_text = ""
                document_text += f"\nPath to the mentioned document: {question.filename}"
            question.content += document_text
        question.filename = None
        logger.info(f"Question: {question.content}")
        return [question] if not image else [question, image]


def print_last_line(python_code: str) -> str:
    lines = python_code.splitlines()
    if " = " in lines[-1]:
        name = lines[-1].split("=")[0].strip()
        lines.append(f"print({name})")
    else:
        lines[-1] = f"print({lines[-1]})"
    return "\n".join(lines)

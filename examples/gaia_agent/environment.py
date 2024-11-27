import logging
import os
import shutil

from tapeagents.container_executor import CodeBlock, ContainerExecutor
from tapeagents.core import Action
from tapeagents.environment import CodeExecutionResult, Environment, ExecuteCode
from tapeagents.tools.calculator import calculate
from tapeagents.tools.document_converters import pdf_to_images
from tapeagents.tools.python_interpreter import run_python_code
from tapeagents.tools.simple_browser import SimpleTextBrowser
from tapeagents.utils import FatalError

from .steps import (
    ActionExecutionFailure,
    CalculationResultObservation,
    CodeResultObservation,
    ConvertFactAction,
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
                    case ExecuteCode():
                        assert self.code_sandbox is not None, "Code sandbox is not provided"
                        result = self.code_sandbox.execute_code_blocks(action.code)
                        obs = CodeExecutionResult(result=result)
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
        logger.info(f"Question: {task['Question']}")
        steps: list[GaiaStep] = [GaiaQuestion.from_task(task)]
        filename: str | None = steps[0].filename  # type: ignore
        if filename:
            name, ext = filename.rsplit(".", maxsplit=1)
            ext = ext.lower()
            if ext == "zip":
                folder_name = name
                os.makedirs(folder_name, exist_ok=True)
                shutil.unpack_archive(filename, folder_name)
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
                steps.append(ImageObservation(image_path=filename, image_caption="Attached image"))
                document_text = ""
            else:
                attach_doc_text = True
                if ext == "pdf" and self.image_observations:
                    images, total_pages = pdf_to_images(filename)
                    if total_pages <= 3:
                        attach_doc_text = False
                    for i, img_path in enumerate(images):
                        steps.append(ImageObservation(image_path=img_path, image_caption=f"PDF page {i+1}"))
                if attach_doc_text:
                    content = self.browser.get_whole_document(filename)
                    document_text = f"\n\n{ext.upper()} document content:\n{content}\n"
                    if len(document_text) > max_doc_length:
                        document_text = ""
                    document_text += f"\nPath to the mentioned document: {filename}"
                else:
                    document_text = "\nDocument pages attached as images below"
            steps[0].content += document_text  # type: ignore
        steps[0].filename = None  # type: ignore
        return steps


def print_last_line(python_code: str) -> str:
    lines = python_code.splitlines()
    if "print(" in lines[-1]:
        return python_code
    if " = " in lines[-1]:
        name = lines[-1].split("=")[0].strip()
        lines.append(f"print({name})")
    else:
        lines[-1] = f"print({lines[-1]})"
    return "\n".join(lines)

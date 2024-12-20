import logging
import os
import shutil

from pdf2image import convert_from_path

from tapeagents.core import Action
from tapeagents.environment import CodeExecutionResult, Environment, ExecuteCode
from tapeagents.steps import WatchVideoAction
from tapeagents.tools.calculator import calculate
from tapeagents.tools.container_executor import CodeBlock, CommandLineCodeResult, ContainerExecutor
from tapeagents.tools.media_reader import get_video_observation
from tapeagents.tools.python_interpreter import run_python_code
from tapeagents.tools.simple_browser import SimpleTextBrowser
from tapeagents.utils import FatalError

from .steps import (
    ActionExecutionFailure,
    CalculationResultObservation,
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
        attachment_dir: str,
        code_sandbox: ContainerExecutor | None = None,
        image_observations: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        self.code_sandbox = code_sandbox
        self.image_observations = image_observations
        self.attachment_dir = attachment_dir
        self.browser = SimpleTextBrowser(**kwargs)

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
                        else:
                            query = action.query

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
                        result = calculate(
                            action.expression,
                            {"value": action.fact_value, action.original_fact_name: action.fact_value},
                        )
                        tape = tape.append(CalculationResultObservation(name=action.converted_fact_name, result=result))
                    case UseCalculatorAction():
                        result = calculate(action.expression, action.facts or {})
                        tape = tape.append(CalculationResultObservation(name=action.fact_name, result=result))
                    case PythonCodeAction():
                        code = add_print_to_last_line(action.code)
                        if self.code_sandbox is None:
                            obs = self.run_restricted_python(code)
                        else:
                            result = self.code_sandbox.execute_code_blocks([CodeBlock(code=code, language="python")])
                            result.output = result.output[:1000].strip()
                            obs = CodeExecutionResult(result=result)
                        tape = tape.append(obs)
                    case ExecuteCode():
                        if self.code_sandbox is None:
                            obs = self.run_restricted_python(action.code[0].code)
                        else:
                            for i in range(len(action.code)):
                                action.code[i].code = add_print_to_last_line(action.code[i].code)
                            result = self.code_sandbox.execute_code_blocks(action.code)
                            result.output = result.output[:1000].strip()
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

    def run_restricted_python(self, code: str) -> CodeExecutionResult:
        logger.warning(f"Code sandbox is not provided, running code locally!\n{code}")
        result, stdout, stderr = run_python_code(code, {})
        output = f"{result[:1000].strip()}\n\nstdout:\n{stdout}\n\nstderr:\n{stderr}"
        return CodeExecutionResult(result=CommandLineCodeResult(output=output, exit_code=0 if not stderr else 1))

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


def add_print_to_last_line(python_code: str) -> str:
    lines = python_code.splitlines()
    if "print(" in lines[-1]:
        return python_code
    if " = " in lines[-1]:
        name = lines[-1].split("=")[0].strip()
        lines.append(f"print({name})")
    else:
        lines[-1] = f"print({lines[-1]})"
    return "\n".join(lines)


def pdf_to_images(filename: str, n_pages: int = 3):
    images = []
    for i, image in enumerate(convert_from_path(filename)):
        page_index = i + 1
        page_fname = filename[:-4] + f"_{page_index}.png"
        if os.path.exists(page_fname):
            images.append(page_fname)
            continue
        image.save(page_fname)
        images.append(page_fname)
    return images[:n_pages], len(images)

from typing import Literal

from pydantic import Field

from tapeagents.core import Action, Observation
from tapeagents.tools.base import Tool
from tapeagents.tools.converters import FileConversionException, FileConverter, UnsupportedFormatException


def read_document(path: str) -> tuple[str, str | None]:
    try:
        text = ""
        error = None
        text = FileConverter().convert(path).text_content
    except UnsupportedFormatException as e:
        error = f"Failed to read document {path}: {e}"
    except FileConversionException as e:
        error = f"Failed to read document {path}: {e}"
    except Exception as e:
        error = f"Failed to read document {path}: {e}"
    return text, error


class DocumentObservation(Observation):
    kind: Literal["document_observation"] = "document_observation"
    text: str
    error: str | None = None


class ReadLocalDocumentAction(Action):
    """
    Action that loads the document, file or image and converts it to Markdown.
    """

    kind: Literal["read_local_document_action"] = "read_local_document_action"
    path: str = Field(description="path of the document")


class DocumentReader(Tool):
    """
    Tool to read a document and convert it to Markdown.
    """

    action: type[Action] = ReadLocalDocumentAction
    observation: type[Observation] = DocumentObservation

    def execute_action(self, action: ReadLocalDocumentAction) -> DocumentObservation:
        text, error = read_document(action.path)
        return DocumentObservation(text=text, error=error)

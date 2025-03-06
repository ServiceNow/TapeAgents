from typing import Literal, Optional

from pydantic import Field

from tapeagents.core import Action, Observation
from tapeagents.tools.base import Tool
from tapeagents.tools.converters import (
    FileConversionException,
    FileConverter,
    PdfConverter,
    PdfMinerConverter,
    UnsupportedFormatException,
)


def read_document(
    path: str, preferred_pdf_converter: Optional[type[PdfConverter | PdfMinerConverter]] = PdfConverter
) -> tuple[str, str | None]:
    """Read a document, file or image and and convert it to Markdown."""
    try:
        text = ""
        error = None
        text = FileConverter(preferred_pdf_converter=preferred_pdf_converter).convert(path).text_content
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
    preferred_pdf_converter: Optional[type[PdfConverter | PdfMinerConverter]] = PdfConverter

    def execute_action(self, action: ReadLocalDocumentAction) -> DocumentObservation:
        text, error = read_document(action.path, self.preferred_pdf_converter)
        return DocumentObservation(text=text, error=error)

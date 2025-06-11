from pathlib import Path
from typing import Literal, Optional

from pydantic import Field

from tapeagents.core import Action, Observation
from tapeagents.tools.base import Tool
from tapeagents.tools.converters import (
    FileConversionException,
    FileConverter,
    FileConverterOptions,
    UnsupportedFormatException,
)


def read_document(path: str, file_converter_options: Optional[FileConverterOptions] = None) -> tuple[str, str | None]:
    """Read a document, file or image and and convert it to Markdown."""
    try:
        text = ""
        error = None
        text = FileConverter(file_converter_options=file_converter_options).convert(path).text_content
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
    source: str | None = None  # optional for tape backwards compatibility
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
    file_converter_options: Optional[FileConverterOptions] = None
    workspace_directory: str | None = None

    def execute_action(self, action: ReadLocalDocumentAction) -> DocumentObservation:
        if self.workspace_directory is not None:
            # Resolve the path safely within the workspace directory
            workspace_dir = Path(self.workspace_directory).resolve()
            path = Path(action.path)
            abs_path = (workspace_dir / path).resolve() if not path.is_absolute() else path.resolve()
            try:
                abs_path.relative_to(workspace_dir)
            except ValueError:
                return DocumentObservation(
                    text="", error="Access to files outside the workspace directory is not allowed"
                )
            workspace_path = abs_path
        else:
            workspace_path = Path(action.path)
        if not workspace_path.exists():
            return DocumentObservation(text="", error=f"File {action.path} not found in the workspace")
        text, error = read_document(str(workspace_path), self.file_converter_options)
        return DocumentObservation(text=text, source=str(action.path), error=error)


class ListDocumentsAction(Action):
    """
    Action that lists all documents in the workspace directory.
    """

    kind: Literal["list_documents_action"] = "list_documents_action"  # type: ignore


class ListDocumentsObservation(Observation):
    """
    Observation that lists all documents in the workspace directory.
    """

    kind: Literal["list_documents_observation"] = "list_documents_observation"  # type: ignore
    documents: list[str] = Field(description="List of documents in the workspace directory")
    error: str | None = None


class ListDocuments(Tool):
    """
    Tool to list all documents in the workspace directory.
    """

    action: type[Action] = ListDocumentsAction
    observation: type[Observation] = ListDocumentsObservation
    workspace_directory: str

    def execute_action(self, action: ListDocumentsAction) -> ListDocumentsObservation:
        try:
            documents = self.list_documents()
            return ListDocumentsObservation(documents=documents)
        except Exception as e:
            return ListDocumentsObservation(documents=[], error=str(e))

    def list_documents(self) -> list[str]:
        """
        List all documents in the workspace directory.
        """
        import os

        documents = []
        os.makedirs(self.workspace_directory, exist_ok=True)
        for root, _, files in os.walk(self.workspace_directory):
            for file in files:
                abs_path = os.path.join(root, file)
                rel_path = os.path.relpath(abs_path, self.workspace_directory)
                documents.append(rel_path)
        return documents

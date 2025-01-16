from typing import Any, Literal

from pydantic import Field

from tapeagents.core import Action, Observation
from tapeagents.tools.base import Tool
from tapeagents.tools.converters import FileConverter


class DocumentObservation(Observation):
    kind: Literal["document_observation"] = "document_observation"
    text: str


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
    kwargs: dict = Field(default_factory=dict)

    def model_post_init(self, __context: Any) -> None:
        self._mdconvert = FileConverter()

    def execute_action(self, action: ReadLocalDocumentAction) -> DocumentObservation:
        res = self._mdconvert.convert_local(action.path, **self.kwargs)
        return DocumentObservation(text=res.text_content)

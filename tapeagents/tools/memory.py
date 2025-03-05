from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal

from pydantic import BaseModel, Field, TypeAdapter

from tapeagents.core import Action, Observation
from tapeagents.tools.base import StatefulTool


class GetMemoryTitlesAction(Action):
    """
    Action to get the titles for all the memories.
    """

    kind: Literal["get_memory_titles_action"] = "get_memory_titles_action"


class GetMemoryAction(Action):
    """
    Action to get a specific memory
    """

    kind: Literal["get_memory_action"] = "get_memory_action"
    key: str = Field(description="A descriptive key for the memory")


class UpsertMemoryAction(Action):
    """
    Action to insert or update a memory
    """

    kind: Literal["upsert_memory_action"] = "upsert_memory_action"
    key: str = Field(description="A descriptive key for the memory")
    title: str = Field(description="title for the memory")
    data: str = Field(description="data for the memory")


class MemoryObservation(Observation):
    kind: Literal["memory_observation"] = "memory_observation"
    key: str = Field(description="A descriptive key for the memory")
    title: str = Field(description="title for the memory")
    data: str = Field(description="data for the memory")


class MemoryTitle(BaseModel):
    key: str = Field(description="A descriptive key for the memory")
    title: str = Field(description="title for the memory")


class MemoryTitlesObservation(Observation):
    kind: Literal["memory_titles_observation"] = "memory_titles_observation"
    titles: List[MemoryTitle] = Field(description="titles and keys for the memories")


class Memory(BaseModel):
    key: str
    title: str
    data: str


class MemoryStorage(ABC):
    @abstractmethod
    def get(self, key: str) -> Memory:
        pass

    @abstractmethod
    def get_all(self) -> List[Memory]:
        pass

    @abstractmethod
    def upsert(self, memory: Memory):
        pass

    @abstractmethod
    def close():
        pass


class InMemoryStorage(MemoryStorage):
    _storage: Dict[str, Memory]

    def __init__(self):
        self._storage: Dict[str, Memory] = {}

    def get(self, key: str) -> Memory:
        value = self._storage.get(key)
        if value is None:
            raise ValueError(f"No memory found for key {key}")
        return TypeAdapter(Memory).validate_python(value)

    def get_all(self) -> List[Memory]:
        memories = []
        for value in self._storage.values():
            memories.append(TypeAdapter(Memory).validate_python(value))
        return memories

    def upsert(self, memory: Memory):
        self._storage[memory.key] = memory

    def close(self):
        self._storage = {}


class LocalMemoryStorage(InMemoryStorage):
    _local_storage_file = Path(__file__).resolve().parent.parent.parent / Path("memory_storage.json")

    def __init__(self):
        try:
            with open(self._local_storage_file, "rb") as file:
                self._storage = TypeAdapter(Dict[str, Memory]).validate_json(file.read())
        except FileNotFoundError:
            self._storage = {}

    def upsert(self, memory: Memory):
        super().upsert(memory)
        # TODO: Add debouncing to avoid writing to disk on every upsert
        with open(self._local_storage_file, "wb") as file:
            file.write(TypeAdapter(Dict[str, Memory]).dump_json(self._storage, indent=2))


class MemoryTool(StatefulTool):
    """
    Tool that manages memory.
    Can store memories.
    Can list the keys and titles of all the memories that are stored for later retrieval.
    """

    actions: tuple[type[Action], ...] = (GetMemoryTitlesAction, GetMemoryAction, UpsertMemoryAction)
    observations: tuple[type[Observation], ...] = (MemoryObservation, MemoryTitlesObservation)
    _action_map: dict[type[Action], Callable] = {}
    _memory_storage: MemoryStorage

    def model_post_init(self, __context: Any):
        self._memory_storage = LocalMemoryStorage()
        self._action_map = {
            GetMemoryTitlesAction: self.get_all_memories,
            GetMemoryAction: self.get_memory,
            UpsertMemoryAction: self.upsert_memory,
        }

    def get_all_memories(self, action: GetMemoryTitlesAction) -> MemoryTitlesObservation:
        memories = self._memory_storage.get_all()
        return MemoryTitlesObservation(titles=[MemoryTitle(key=memory.key, title=memory.title) for memory in memories])

    def get_memory(self, action: GetMemoryAction) -> MemoryObservation:
        memory = self._memory_storage.get(action.key)
        return MemoryObservation(key=memory.key, title=memory.title, data=memory.data)

    def upsert_memory(self, action: UpsertMemoryAction) -> MemoryObservation:
        self._memory_storage.upsert(Memory(key=action.key, title=action.title, data=action.data))
        return MemoryObservation(key=action.key, title=action.title, data=action.data)

    def execute_action(self, action: Action) -> MemoryObservation | MemoryTitlesObservation:
        action_type = type(action)
        if action_type in self._action_map:
            return self._action_map[action_type](action)
        raise ValueError(f"Unknown action: {action_type}")

    def close(self) -> None:
        self._memory_storage.close()

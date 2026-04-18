from typing import Any, Protocol


class Dataset(Protocol):
    name: str
    description: str

    def load(self, regions: list[str]) -> Any: ...

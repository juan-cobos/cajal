from typing import Any, Protocol


class Dataset(Protocol):
    name: str
    description: str
    regions: list[str]

    def load(self) -> Any: ...

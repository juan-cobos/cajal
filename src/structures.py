import json
from pathlib import Path

import requests

from src.urls import STRUCTURES as _URL

_PATH = Path(__file__).parent.parent / "data/structures.json"


def _walk(node: dict, result: list):
    result.append({k: v for k, v in node.items() if k != "children"})
    for child in node.get("children", []):
        _walk(child, result)


def _load() -> list[dict]:
    if not _PATH.exists():
        _PATH.parent.mkdir(parents=True, exist_ok=True)
        structures = []
        for root in requests.get(_URL).json()["msg"]:
            _walk(root, structures)
        _PATH.write_text(json.dumps(structures))
    return json.loads(_PATH.read_text())


STRUCTURES = _load()
ACRONYM_TO_ID: dict[str, int] = {s["acronym"]: s["id"] for s in STRUCTURES}
ID_TO_ACRONYM: dict[int, str] = {s["id"]: s["acronym"] for s in STRUCTURES}

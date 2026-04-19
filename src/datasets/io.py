import asyncio
import json
from pathlib import Path

import aiohttp
import requests

from src.urls import ABC_MANIFEST

_MANIFEST_PATH = Path(__file__).parent.parent.parent / "data/manifest.json"


async def download(session: aiohttp.ClientSession, url: str, dest: Path) -> None:
    async with session.get(url, timeout=aiohttp.ClientTimeout()) as resp:
        resp.raise_for_status()
        dest.write_bytes(await resp.read())


async def ensure_files(urls_and_paths: list[tuple[str, Path]]) -> None:
    missing = [(url, path) for url, path in urls_and_paths if not path.exists()]
    if not missing:
        return
    async with aiohttp.ClientSession() as session:
        await asyncio.gather(*[download(session, url, path) for url, path in missing])


def _load_expr_index() -> dict[str, str]:
    _MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not _MANIFEST_PATH.exists():
        _MANIFEST_PATH.write_bytes(requests.get(ABC_MANIFEST).content)
    manifest = json.loads(_MANIFEST_PATH.read_text())
    urls = {}
    for v in manifest["file_listing"].values():
        try:
            for dataset, dv in v["expression_matrices"].items():
                urls[dataset] = dv["log2"]["files"]["h5ad"]["url"]
        except (KeyError, TypeError):
            pass
    return urls


EXPR_INDEX: dict[str, str] = _load_expr_index()

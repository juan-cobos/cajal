import asyncio
from pathlib import Path

import aiohttp
import polars as pl

_BASE_URL = (
    "http://api.brain-map.org/api/v2/data/query.json"
    "?criteria=service::mouse_connectivity_injection_structure"
)
_UNIONIZE_URL = "http://api.brain-map.org/api/v2/data/query.json?criteria=model::ProjectionStructureUnionize"
_CACHE_DIR = Path(__file__).parent.parent.parent / "data/unionizes"


async def _fetch_region(
    session: aiohttp.ClientSession,
    region: str,
    mouse_line: str | None,
    primary_injection: bool,
) -> tuple[str, list[int]]:
    url = (
        f"{_BASE_URL}[injection_structures$eq{region}]"
        f"{'[transgenic_lines$eq' + mouse_line + ']' if mouse_line else ''}"
        f"[primary_structure_only$eq{'true' if primary_injection else 'false'}]"
    )
    async with session.get(url) as resp:
        result = await resp.json(content_type=None)
    if not result.get("success"):
        print(f"Query failed for {region}: {result.get('msg')}")
        return region, []
    entries = result["msg"]
    print(f"Found {len(entries)} experiments in {region}")
    return region, [e["id"] for e in entries]


def _find_experiments(
    regions: list[str],
    mouse_line: str | None = None,
    primary_injection: bool = True,
) -> dict[str, list[int]]:
    async def _gather() -> dict[str, list[int]]:
        async with aiohttp.ClientSession() as session:
            results = await asyncio.gather(
                *[_fetch_region(session, r, mouse_line, primary_injection) for r in regions]
            )
        return dict(results)

    return asyncio.run(_gather())


async def _fetch_unionizes(session: aiohttp.ClientSession, url: str) -> list[dict]:
    async with session.get(url, timeout=aiohttp.ClientTimeout()) as resp:
        result = await resp.json(content_type=None)
    if not result.get("success"):
        raise RuntimeError(result.get("msg"))
    return result["msg"]


def _get_structure_unionizes(
    experiments: dict[str, list[int]],
    is_injection: bool = True,
    structure_ids: list[int] | None = None,
    hemisphere_ids: list[int] | None = None,
    num_rows: int | str = "all",
) -> pl.DataFrame:
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)

    frames, missing = [], {}
    for region, eids in experiments.items():
        for eid in eids:
            path = _CACHE_DIR / f"{region}_{eid}.parquet"
            if path.exists():
                frames.append(pl.read_parquet(path).with_columns(pl.lit(region).alias("input_region")))
            else:
                missing.setdefault(region, []).append(eid)

    if missing:
        async def _gather_unionizes() -> list[pl.DataFrame]:
            async with aiohttp.ClientSession() as session:
                results = await asyncio.gather(*[
                    _fetch_unionizes(
                        session,
                        f"{_UNIONIZE_URL}"
                        f"[section_data_set_id$in{','.join(str(i) for i in eids)}]"
                        f"&num_rows={num_rows}&count=false",
                    )
                    for eids in missing.values()
                ])
            return [
                pl.DataFrame(rows).with_columns(pl.lit(region).alias("input_region"))
                for region, rows in zip(missing.keys(), results)
            ]

        for region, df in zip(missing.keys(), asyncio.run(_gather_unionizes())):
            for eid in missing[region]:
                subset = df.filter(pl.col("section_data_set_id") == eid)
                if not subset.is_empty():
                    subset.write_parquet(_CACHE_DIR / f"{region}_{eid}.parquet")
            frames.append(df)

    if not frames:
        return pl.DataFrame()

    df = pl.concat(frames)
    df = df.filter(pl.col("is_injection") == is_injection)
    if structure_ids is not None:
        df = df.filter(pl.col("structure_id").is_in(structure_ids))
    if hemisphere_ids is not None:
        df = df.filter(pl.col("hemisphere_id").is_in(hemisphere_ids))

    return df


class ConnectivityDataset:
    name = "connectivity"
    description = "Allen Mouse Brain connectivity experiments and structure unionizes grouped by injection region."

    def load(self, regions: list[str]) -> pl.DataFrame:
        experiments = _find_experiments(regions)
        return _get_structure_unionizes(experiments)

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Literal

import polars as pl

_CSV_PATH = Path(__file__).parent.parent.parent / "data/pfcmap/units.csv"


@dataclass
class Unit:
    uid: int
    u: float
    v: float
    AP: float
    DV: float
    ML: float
    region: str
    chan: float
    roi: int
    wavequality: float
    quality: int
    B_mean: float
    M_mean: float
    rate_mean: float
    utype: Literal["nw", "ww", "na"]  # wide-width, narrow-width, unclassified
    area: str
    layer: str
    dataset: str
    task: str
    recid: str
    bmu: int
    cluster: int
    passes_filters: bool


class NeuralActivityDataset:
    name = "neural_activity"
    description = "Single-neuron electrophysiology activity metrics from pfcmap (Carlen/IBL datasets)."

    def load(
        self,
        regions: list[str] | None = None,
        utypes: list[Literal["nw", "ww", "na"]] | None = None,
        passes_filters: bool | None = None,
    ) -> pl.DataFrame:
        df = pl.read_csv(_CSV_PATH)
        # TODO: maybe I can write as pl.group_by().agg()...
        if utypes is not None:
            df = df.filter(pl.col("utype").is_in(utypes))
        if regions is not None:
            df = df.filter(
                pl.col("area").is_in(regions) | pl.col("region").is_in(regions)
            )
        if passes_filters is not None:
            df = df.filter(pl.col("passes_filters") == passes_filters)
        return df

    def load_units(
        self,
        regions: list[str] | None = None,
        utypes: list[Literal["nw", "ww", "na"]] | None = None,
        passes_filters: bool | None = None,
    ) -> list[Unit]:
        df = self.load(regions=regions, utypes=utypes, passes_filters=passes_filters)
        field_names = {f.name for f in fields(Unit)}
        cols = [c for c in df.columns if c in field_names]
        return [
            Unit(**{k: row[k] for k in cols})
            for row in df.select(cols).iter_rows(named=True)
        ]


if __name__ == "__main__":
    ds = NeuralActivityDataset()
    df = ds.load(utypes=["ww"], passes_filters=True)
    print(df.columns)
    from collections import Counter
    import pprint

    all_regions = set(df["region"])
    df_prelimbic = df.filter(pl.col("region").is_in(["PL5", "PL6a"]))
    print(f"n_units in prelimbic: {len(df_prelimbic)}")

    pfc_areas = ["PL", "ILA", "ACAd", "ACAv", "ORBm", "ORBl", "ORBvl", "FRP"]
    df_pfc = df.filter(pl.col("area").is_in(pfc_areas))
    print(f"n_pfc_units: {len(df_pfc)}")

    k = 10
    top_regions = (
        df_pfc.group_by("region")
        .len()
        .sort("len", descending=True)
        .head(k)
    )
    print(f"top {k} pfc regions:")
    print(top_regions)

import asyncio
from pathlib import Path

import anndata as ad
import polars as pl

from src.datasets.io import EXPR_INDEX, ensure_files
from src.urls import SCRNA_METADATA

_CACHE_DIR = Path(__file__).parent.parent.parent / "data/scrna"

_FILE_COL = "feature_matrix_label"
_REGION_COL = "region_of_interest_acronym"


def _load_metadata(regions: list[str]) -> pl.DataFrame:
    meta_path = _CACHE_DIR / "cell_metadata.parquet"
    asyncio.run(ensure_files([(SCRNA_METADATA, meta_path)]))
    filtered = pl.read_parquet(meta_path).filter(pl.col(_REGION_COL).is_in(regions))
    if filtered.is_empty():
        raise ValueError(
            f"No cells found for regions {regions} in column '{_REGION_COL}'"
        )
    return filtered


class ScRNADataset:
    name = "scrna"
    description = "Allen Brain Cell Atlas single-cell RNA sequencing data."

    def load(self, regions: list[str]) -> ad.AnnData:
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)

        meta = _load_metadata(regions)
        files = meta[_FILE_COL].unique().to_list()
        paths = [_CACHE_DIR / f"{f}.h5ad" for f in files]

        missing_files = [f for f in files if f not in EXPR_INDEX]
        if missing_files:
            raise ValueError(f"No expression matrix URL found for: {missing_files}")

        asyncio.run(ensure_files(list(zip([EXPR_INDEX[f] for f in files], paths))))

        cell_labels = set(meta["cell_label"].to_list())
        parts = []
        for path in paths:
            adata = ad.read_h5ad(path, backed="r")
            parts.append(adata[adata.obs_names.isin(cell_labels)].to_memory())

        return ad.concat(parts) if len(parts) > 1 else parts[0]


if __name__ == "__main__":
    ds = ScRNADataset()
    adata = ds.load(["PL5", "PL6a"])

    print(adata)
    print(adata.obs.head())

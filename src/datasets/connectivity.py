import polars as pl

from src.experiments import find_experiments, get_structure_unionizes


class ConnectivityDataset:
    name = "connectivity"
    description = "Allen Mouse Brain connectivity experiments and structure unionizes grouped by injection region."

    def load(self, regions: list[str]) -> pl.DataFrame:
        experiments = find_experiments(regions)
        return get_structure_unionizes(experiments)

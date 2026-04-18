import polars as pl

from src.experiments import find_experiments, get_structure_unionizes


class ConnectivityDataset:
    name = "connectivity"
    description = "Allen Mouse Brain connectivity experiments and structure unionizes grouped by injection region."

    def __init__(self, regions: list[str]):
        self.regions = regions

    def load(self) -> pl.DataFrame:
        experiments = find_experiments(self.regions)
        return get_structure_unionizes(experiments)

from src.datasets.connectivity import ConnectivityDataset
from src.datasets.scrna import ScRNADataset

_REGISTRY: dict[str, type] = {
    ConnectivityDataset.name: ConnectivityDataset,
    ScRNADataset.name: ScRNADataset,
}


class DataLoader:
    """Load one or more datasets grouped by brain region.

    Parameters
    ----------
    regions : list of str
        Brain region acronyms (e.g. ``['VTA', 'DR']``).
    """

    def __init__(self, regions: list[str]):
        self.regions = regions

    def list_datasets(self) -> list[str]:
        return list(_REGISTRY.keys())

    def describe(self, name: str) -> str:
        return _REGISTRY[name].description

    def load(self, name: str | list[str]):
        if isinstance(name, list):
            return {n: _REGISTRY[n](self.regions).load() for n in name}
        return _REGISTRY[name](self.regions).load()

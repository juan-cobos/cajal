from src.datasets.connectivity import ConnectivityDataset
from src.datasets.merfish import MerfishDataset
from src.datasets.neural_activity import NeuralActivityDataset
from src.datasets.scrna import ScRNADataset
from src.datasets.base import Dataset

_REGISTRY: dict[str, Dataset] = {
    ConnectivityDataset.name: ConnectivityDataset(),
    ScRNADataset.name: ScRNADataset(),
    MerfishDataset.name: MerfishDataset(),
    NeuralActivityDataset.name: NeuralActivityDataset(),
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
            return {n: _REGISTRY[n].load(self.regions) for n in name}
        return _REGISTRY[name].load(self.regions)

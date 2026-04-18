class ScRNADataset:
    name = "scrna"
    description = "Allen Brain Cell Atlas single-cell RNA sequencing data."

    def __init__(self, regions: list[str]):
        self.regions = regions

    def load(self):
        raise NotImplementedError

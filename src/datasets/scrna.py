class ScRNADataset:
    name = "scrna"
    description = "Allen Brain Cell Atlas single-cell RNA sequencing data."

    def load(self, regions: list[str]):
        raise NotImplementedError

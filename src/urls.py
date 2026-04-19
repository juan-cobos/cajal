_ABC_BASE = "https://allen-brain-cell-atlas.s3.us-west-2.amazonaws.com"
_ABC_RELEASE = "20231215"

ABC_MANIFEST = f"{_ABC_BASE}/releases/{_ABC_RELEASE}/manifest.json"

SCRNA_METADATA = f"{_ABC_BASE}/metadata/WMB-10X/{_ABC_RELEASE}/cell_metadata.csv"

MERFISH_METADATA = f"{_ABC_BASE}/metadata/MERFISH-C57BL6J-638850-CCF/{_ABC_RELEASE}/views/cell_metadata_with_parcellation_annotation.csv"

CONNECTIVITY_BASE = "http://api.brain-map.org/api/v2/data/query.json?criteria=service::mouse_connectivity_injection_structure"
CONNECTIVITY_UNIONIZE = "http://api.brain-map.org/api/v2/data/query.json?criteria=model::ProjectionStructureUnionize"

STRUCTURES = "http://api.brain-map.org/api/v2/structure_graph_download/1.json"

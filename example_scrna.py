from pathlib import Path

import anndata
from abc_atlas_access.abc_atlas_cache.abc_project_cache import AbcProjectCache

cache_dir = Path("abc_data")
cache_dir.mkdir(exist_ok=True)
cache = AbcProjectCache.from_s3_cache(cache_dir)

# WMB = Whole Mouse Brain. 10Xv3 is split by broad anatomical division:
# Isocortex, OLF, HPF, CTXsp, STR, PAL, TH, HY, MB, HB, CB.
# MB (midbrain) contains VTA/DR; swap the suffix below to target others.
region = "MB"
directory = "WMB-10Xv3"
file_name = f"WMB-10Xv3-{region}/log2"

h5ad_path = cache.get_data_path(directory=directory, file_name=file_name)
adata = anndata.read_h5ad(h5ad_path, backed="r")
print(f"{region}: {adata.n_obs} cells x {adata.n_vars} genes (backed, nothing loaded yet)")

cell_meta = cache.get_metadata_dataframe(
    directory="WMB-10X", file_name="cell_metadata", dtype={"cell_label": str}
).set_index("cell_label")
cluster_meta = cache.get_metadata_dataframe(
    directory="WMB-taxonomy", file_name="cluster_to_cluster_annotation_membership_pivoted"
)

shared = adata.obs_names.intersection(cell_meta.index)
adata = adata[shared]
adata.obs = adata.obs.join(cell_meta.loc[shared, ["subclass", "class", "cluster_alias"]])

print("\nTop subclasses in", region)
print(adata.obs["subclass"].value_counts().head(10))

genes_of_interest = ["Slc6a3", "Th", "Slc6a4", "Tph2"]  # dopamine + serotonin markers
present = [g for g in genes_of_interest if g in adata.var_names]
if present:
    subset = adata[:, present].to_memory()
    print(f"\nMean log2 expression across {subset.n_obs} cells:")
    print(subset.to_df().mean().round(3))

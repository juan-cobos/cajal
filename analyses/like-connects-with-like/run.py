"""
Hypothesis H32: "Like-connects-with-like" wiring rule in PFC, tested at
the cluster level across three modalities:

1. TRANSCRIPTOMIC: MERFISH cells whose spatial kNN share the same
   transcriptomic cluster/supertype more than expected (within same
   region × layer), indicating micro-architectural transcriptomic
   neighborhoods.

2. FUNCTIONAL: Neural activity neurons in the same ephys cluster are
   spatially closer than neurons in different clusters (within same
   region × layer × utype), indicating functional micro-circuits.

3. CROSS-MODAL: Region-level ephys cluster proportions correlate with
   MERFISH transcriptomic cluster proportions, linking functional and
   transcriptomic cell types.

4. CONNECTIVITY: Regions with similar connectivity profiles have similar
   transcriptomic AND functional cluster compositions.

Branch: like-connects-with-like
Datasets: merfish, neural_activity, connectivity
"""

import glob
import json
import random
from pathlib import Path

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from scipy.spatial.distance import cdist
from scipy.stats import mannwhitneyu, spearmanr, pearsonr, chi2_contingency
from sklearn.decomposition import PCA

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

DATA_PATH = Path(__file__).parent.parent.parent / "data"
OUTPUT_DIR = Path(__file__).parent

L5_REGIONS = ["PL5", "ILA5", "ORBl5", "ORBm5", "ORBvl5"]
L6_REGIONS = ["PL6a", "ILA6a", "ORBl6a", "ORBm6a", "ORBvl6a"]
ALL_REGIONS = L5_REGIONS + L6_REGIONS

print("=" * 60)
print("H32: Like-connects-with-like (cluster level)")
print("=" * 60)

# ============================================================
# PART 1: MERFISH transcriptomic cluster spatial enrichment
# ============================================================
print("\n[1] MERFISH: transcriptomic cluster spatial enrichment...")

adata = ad.read_h5ad(DATA_PATH / "merfish" / "C57BL6J-638850.h5ad")
meta = pl.read_csv(DATA_PATH / "merfish" / "cell_metadata_with_parcellation_annotation.csv")
pfc_cells = meta.filter(pl.col("parcellation_substructure").is_in(ALL_REGIONS))
cell_labels = set(pfc_cells["cell_label"].to_list())
adata = adata[adata.obs_names.isin(cell_labels)].copy()

obs_df = pfc_cells.to_pandas().set_index("cell_label")
common = adata.obs_names.intersection(obs_df.index)
adata = adata[common].copy()
adata.obs["region"] = obs_df.loc[common, "parcellation_substructure"].values
adata.obs["x"] = obs_df.loc[common, "x_reconstructed"].values.astype(float)
adata.obs["y"] = obs_df.loc[common, "y_reconstructed"].values.astype(float)
adata.obs["z"] = obs_df.loc[common, "z_reconstructed"].values.astype(float)
adata.obs["subclass"] = obs_df.loc[common, "subclass"].values
adata.obs["supertype"] = obs_df.loc[common, "supertype"].values
adata.obs["merfish_cluster"] = obs_df.loc[common, "cluster"].values
adata.obs["nt"] = obs_df.loc[common, "neurotransmitter"].values
adata.obs["layer"] = adata.obs["region"].map(lambda x: "5" if x in L5_REGIONS else "6a")

valid = ~(np.isnan(adata.obs["x"].values.astype(float)) |
          np.isnan(adata.obs["y"].values.astype(float)) |
          np.isnan(adata.obs["z"].values.astype(float)))
adata = adata[valid].copy()

coords = np.column_stack([adata.obs["x"].values, adata.obs["y"].values, adata.obs["z"].values])
subclass_labels = adata.obs["subclass"].values
supertype_labels = adata.obs["supertype"].values
cluster_labels = adata.obs["merfish_cluster"].values
region_labels = adata.obs["region"].values
nt_labels = adata.obs["nt"].values
layer_labels = adata.obs["layer"].values

print(f"  {adata.shape[0]} cells with valid coords")

K = 10

def knn_cluster_enrichment(coords, labels, group_labels, k=10, n_sample_per_group=500):
    """For each cell, fraction of kNN sharing its label vs random expectation."""
    actual_fracs = []
    expected_fracs = []
    diffs = []

    unique_groups = np.unique(group_labels)
    for group in unique_groups:
        gmask = group_labels == group
        g_idx = np.where(gmask)[0]
        n_g = len(g_idx)
        if n_g < k + 5:
            continue

        g_coords = coords[g_idx]
        g_labels = labels[g_idx]

        # Pre-compute label fractions in group (expected under random)
        label_counts = {}
        for lb in g_labels:
            label_counts[lb] = label_counts.get(lb, 0) + 1
        n_g_f = float(n_g)

        dists = cdist(g_coords, g_coords, metric="euclidean")

        sample = np.random.choice(n_g, size=min(n_sample_per_group, n_g), replace=False)
        for i in sample:
            lb_i = g_labels[i]
            expected_frac = label_counts.get(lb_i, 0) / n_g_f

            dists_i = dists[i].copy()
            dists_i[i] = np.inf
            knn_idx = np.argsort(dists_i)[:k]
            knn_labels = g_labels[knn_idx]
            actual_frac = np.sum(knn_labels == lb_i) / k

            actual_fracs.append(actual_frac)
            expected_fracs.append(expected_frac)
            diffs.append(actual_frac - expected_frac)

    return np.array(actual_fracs), np.array(expected_fracs), np.array(diffs)

# Test at subclass level (coarsest)
print(f"\n  Subclass-level kNN enrichment (k={K}):")
group_labels_sub = np.array([f"{r}_{l}" for r, l in zip(region_labels, layer_labels)])
act_sub, exp_sub, diff_sub = knn_cluster_enrichment(coords, subclass_labels, group_labels_sub, k=K)
u_sub, p_sub = mannwhitneyu(act_sub, exp_sub, alternative="greater")
mean_enrich_sub = np.mean(diff_sub)
frac_pos_sub = np.mean(diff_sub > 0)
print(f"    n={len(diff_sub)}, actual={np.mean(act_sub):.4f}, expected={np.mean(exp_sub):.4f}, "
      f"Δ={mean_enrich_sub:+.4f}, frac>0={frac_pos_sub:.4f}, MW p={p_sub:.2e}")

# Test at supertype level
print(f"\n  Supertype-level kNN enrichment (k={K}):")
act_st, exp_st, diff_st = knn_cluster_enrichment(coords, supertype_labels, group_labels_sub, k=K)
u_st, p_st = mannwhitneyu(act_st, exp_st, alternative="greater")
mean_enrich_st = np.mean(diff_st)
frac_pos_st = np.mean(diff_st > 0)
print(f"    n={len(diff_st)}, actual={np.mean(act_st):.4f}, expected={np.mean(exp_st):.4f}, "
      f"Δ={mean_enrich_st:+.4f}, frac>0={frac_pos_st:.4f}, MW p={p_st:.2e}")

# Test at cluster level (finest)
print(f"\n  Cluster-level kNN enrichment (k={K}):")
act_cl, exp_cl, diff_cl = knn_cluster_enrichment(coords, cluster_labels, group_labels_sub, k=K)
u_cl, p_cl = mannwhitneyu(act_cl, exp_cl, alternative="greater")
mean_enrich_cl = np.mean(diff_cl)
frac_pos_cl = np.mean(diff_cl > 0)
print(f"    n={len(diff_cl)}, actual={np.mean(act_cl):.4f}, expected={np.mean(exp_cl):.4f}, "
      f"Δ={mean_enrich_cl:+.4f}, frac>0={frac_pos_cl:.4f}, MW p={p_cl:.2e}")

# Within-NT test: only Glut or only GABA cells
print(f"\n  Glut-only subclass kNN enrichment (k={K}):")
glut_mask = nt_labels == "Glut"
glut_group = np.array([f"{r}_{l}_{sc}" for r, l, sc in zip(region_labels[glut_mask], layer_labels[glut_mask], subclass_labels[glut_mask])])
act_glut, exp_glut, diff_glut = knn_cluster_enrichment(
    coords[glut_mask], cluster_labels[glut_mask], glut_group, k=K, n_sample_per_group=80)
u_glut, p_glut = mannwhitneyu(act_glut, exp_glut, alternative="greater")
print(f"    n={len(diff_glut)}, Δ={np.mean(diff_glut):+.4f}, frac>0={np.mean(diff_glut>0):.4f}, p={p_glut:.2e}")

print(f"\n  GABA-only subclass kNN enrichment (k={K}):")
gaba_mask = nt_labels == "GABA"
gaba_group = np.array([f"{r}_{l}_{sc}" for r, l, sc in zip(region_labels[gaba_mask], layer_labels[gaba_mask], subclass_labels[gaba_mask])])
act_gaba, exp_gaba, diff_gaba = knn_cluster_enrichment(
    coords[gaba_mask], cluster_labels[gaba_mask], gaba_group, k=K, n_sample_per_group=80)
u_gaba, p_gaba = mannwhitneyu(act_gaba, exp_gaba, alternative="greater")
print(f"    n={len(diff_gaba)}, Δ={np.mean(diff_gaba):+.4f}, frac>0={np.mean(diff_gaba>0):.4f}, p={p_gaba:.2e}")

# Within-subclass cluster enrichment (the key refined test)
print(f"\n  Within-subclass MERFISH-cluster enrichment (k={K}):")
group_labels_wsc = np.array([f"{r}_{sc}" for r, sc in zip(region_labels, subclass_labels)])
act_wsc, exp_wsc, diff_wsc = knn_cluster_enrichment(
    coords, cluster_labels, group_labels_wsc, k=K, n_sample_per_group=100)
u_wsc, p_wsc = mannwhitneyu(act_wsc, exp_wsc, alternative="greater")
print(f"    n={len(diff_wsc)}, actual={np.mean(act_wsc):.4f}, expected={np.mean(exp_wsc):.4f}, "
      f"Δ={np.mean(diff_wsc):+.4f}, frac>0={np.mean(diff_wsc>0):.4f}, MW p={p_wsc:.2e}")

# ============================================================
# PART 2: Functional cluster spatial enrichment (neural_activity)
# ============================================================
print("\n[2] Neural activity: functional cluster spatial enrichment...")

neural = pl.read_csv(DATA_PATH / "pfcmap" / "units.csv")
neural = neural.filter(pl.col("region").is_in(ALL_REGIONS))
neural = neural.filter(pl.col("utype").is_in(["ww", "nw"]))
neural = neural.filter(pl.col("rate_mean").is_not_null())
neural = neural.filter(pl.col("B_mean").is_not_null())
print(f"  {len(neural)} neurons")

ephys_coords = np.column_stack([
    neural["AP"].to_numpy(),
    neural["DV"].to_numpy(),
    neural["ML"].to_numpy(),
])
ephys_clusters = neural["cluster"].to_numpy()
ephys_utypes = neural["utype"].to_numpy()
ephys_regions = neural["region"].to_numpy()
ephys_layers = neural["layer"].to_numpy()

# Test: within each region×layer×utype, are same-cluster neurons closer?
print(f"\n  Within-region spatial distance by same vs different cluster:")

same_cluster_dists = []
diff_cluster_dists = []

for region in ALL_REGIONS:
    for layer in ["5", "6a"]:
        for utype in ["ww", "nw"]:
            rmask = (ephys_regions == region) & (ephys_layers == layer) & (ephys_utypes == utype)
            r_idx = np.where(rmask)[0]
            n_r = len(r_idx)
            if n_r < 10:
                continue

            r_coords = ephys_coords[r_idx]
            r_clusters = ephys_clusters[r_idx]

            # Sample pairs to avoid O(n²)
            n_pairs = min(5000, n_r * (n_r - 1) // 2)
            for _ in range(n_pairs):
                i, j = np.random.choice(n_r, size=2, replace=False)
                d = np.linalg.norm(r_coords[i] - r_coords[j])
                if r_clusters[i] == r_clusters[j]:
                    same_cluster_dists.append(d)
                else:
                    diff_cluster_dists.append(d)

same_cluster_dists = np.array(same_cluster_dists)
diff_cluster_dists = np.array(diff_cluster_dists)

if len(same_cluster_dists) > 0 and len(diff_cluster_dists) > 0:
    u_func, p_func = mannwhitneyu(same_cluster_dists, diff_cluster_dists, alternative="less")
    print(f"    Same-cluster dist: {np.mean(same_cluster_dists):.1f} ± {np.std(same_cluster_dists):.1f} μm (n={len(same_cluster_dists)})")
    print(f"    Diff-cluster dist: {np.mean(diff_cluster_dists):.1f} ± {np.std(diff_cluster_dists):.1f} μm (n={len(diff_cluster_dists)})")
    print(f"    Same < diff? MW p={p_func:.2e}")
else:
    p_func = 1.0
    print("    Insufficient data")

# ============================================================
# PART 3: Cross-modal cluster composition correlation
# ============================================================
print("\n[3] Cross-modal: ephys cluster proportions ↔ MERFISH cluster proportions per region...")

merfish_comp = {}
for region in ALL_REGIONS:
    rmask = region_labels == region
    r_sub = subclass_labels[rmask]
    r_nt = nt_labels[rmask]
    n_r = int(rmask.sum())

    sub_counts = {}
    for s, nt in zip(r_sub, r_nt):
        key = f"{s}_{nt}"
        sub_counts[key] = sub_counts.get(key, 0) + 1
    for key in sub_counts:
        sub_counts[key] /= n_r
    merfish_comp[region] = sub_counts

ephys_comp = {}
for region in ALL_REGIONS:
    sub = neural.filter(pl.col("region") == region)
    n_sub = len(sub)
    cl_counts = {}
    for row in sub.iter_rows(named=True):
        key = f"C{row['cluster']}_{row['utype']}"
        cl_counts[key] = cl_counts.get(key, 0) + 1
    for key in cl_counts:
        cl_counts[key] /= n_sub
    ephys_comp[region] = cl_counts

# Find common keys across all regions
merfish_keys = set()
for v in merfish_comp.values():
    merfish_keys |= set(v.keys())
ephys_keys = set()
for v in ephys_comp.values():
    ephys_keys |= set(v.keys())

# Build matrices
mf_matrix = np.zeros((len(ALL_REGIONS), len(merfish_keys)))
mf_keys = sorted(merfish_keys)
for i, region in enumerate(ALL_REGIONS):
    for j, key in enumerate(mf_keys):
        mf_matrix[i, j] = merfish_comp[region].get(key, 0)

ep_matrix = np.zeros((len(ALL_REGIONS), len(ephys_keys)))
ep_keys = sorted(ephys_keys)
for i, region in enumerate(ALL_REGIONS):
    for j, key in enumerate(ep_keys):
        ep_matrix[i, j] = ephys_comp[region].get(key, 0)

# Filter to columns with variance
mf_var = mf_matrix.std(axis=0) > 0
ep_var = ep_matrix.std(axis=0) > 0
mf_matrix_f = mf_matrix[:, mf_var]
ep_matrix_f = ep_matrix[:, ep_var]

print(f"  MERFISH composition: {mf_matrix_f.shape[1]} non-constant subclass×NT fractions")
print(f"  Ephys composition: {ep_matrix_f.shape[1]} non-constant cluster×utype fractions")

# Mantel-like test: are composition distance matrices correlated?
def corr_distance(m):
    n = m.shape[0]
    d = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            r, _ = pearsonr(m[i], m[j])
            d[i, j] = 1 - r
            d[j, i] = 1 - r
    return d

mf_dist = corr_distance(mf_matrix_f)
ep_dist = corr_distance(ep_matrix_f)

d1 = mf_dist[np.triu_indices(len(ALL_REGIONS), k=1)]
d2 = ep_dist[np.triu_indices(len(ALL_REGIONS), k=1)]
r_cross, p_cross = pearsonr(d1, d2)

n_perm = 10000
perm_rs = []
for _ in range(n_perm):
    idx = np.random.permutation(len(d1))
    pr, _ = pearsonr(d1, d2[idx])
    perm_rs.append(pr)
p_cross_perm = (np.sum(np.abs(perm_rs) >= np.abs(r_cross)) + 1) / (n_perm + 1)

layer_vec = np.array([0.0 if r in L5_REGIONS else 1.0 for r in ALL_REGIONS])
layer_dist = np.abs(layer_vec[:, None] - layer_vec[None, :])
ld1 = layer_dist[np.triu_indices(len(ALL_REGIONS), k=1)]

from scipy.stats import rankdata
d1_r = rankdata(d1)
d2_r = rankdata(d2)
ld1_r = rankdata(ld1)
X = np.column_stack([np.ones(len(ld1_r)), ld1_r])
res1 = d1_r - X @ np.linalg.lstsq(X, d1_r, rcond=None)[0]
res2 = d2_r - X @ np.linalg.lstsq(X, d2_r, rcond=None)[0]
r_cross_partial, p_cross_partial = pearsonr(res1, res2)

print(f"  Composition Mantel: r={r_cross:+.4f}, p_perm={p_cross_perm:.4f}")
print(f"  Partial (layer ctrl): r={r_cross_partial:+.4f}, p={p_cross_partial:.4f}")

# ============================================================
# PART 4: Connectivity clusters ↔ transcriptomic/functional composition
# ============================================================
print("\n[4] Connectivity clusters ↔ composition...")

conn_matrix = {}
for region in ALL_REGIONS:
    files = glob.glob(str(DATA_PATH / "connectivity" / f"{region}_*.parquet"))
    all_dfs = []
    for f in files:
        df = pl.read_parquet(f)
        proj = df.filter(pl.col("is_injection") == False)
        all_dfs.append(proj)
    combined = pl.concat(all_dfs)
    avg = combined.group_by("structure_id").agg(
        pl.col("normalized_projection_volume").mean().alias("NPV_mean")
    )
    conn_matrix[region] = dict(zip(avg["structure_id"].to_list(), avg["NPV_mean"].to_list()))

common_targets = sorted(set(conn_matrix[ALL_REGIONS[0]].keys()))
for region in ALL_REGIONS[1:]:
    common_targets = sorted(set(common_targets) & set(conn_matrix[region].keys()))

conn_mat = np.array([[conn_matrix[r].get(t, 0.0) for t in common_targets] for r in ALL_REGIONS])
strong_mask = (conn_mat > 0.1).sum(axis=0) >= 3
conn_strong = conn_mat[:, strong_mask]

conn_dist = corr_distance(conn_strong)
cd = conn_dist[np.triu_indices(len(ALL_REGIONS), k=1)]

r_conn_mf, p_conn_mf = pearsonr(cd, d1)
r_conn_ep, p_conn_ep = pearsonr(cd, d2)

print(f"  Conn ↔ MERFISH composition: r={r_conn_mf:+.4f}, p={p_conn_mf:.4f}")
print(f"  Conn ↔ Ephys composition: r={r_conn_ep:+.4f}, p={p_conn_ep:.4f}")

# ============================================================
# Summary
# ============================================================
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)

trans_sig = p_wsc < 0.001
func_sig = p_func < 0.05 if len(same_cluster_dists) > 0 else False
cross_sig = p_cross_perm < 0.05

print(f"  Transcriptomic like-with-like: {'CONFIRMED' if trans_sig else 'REFUTED'} (within-subclass cluster p={p_wsc:.2e})")
print(f"  Functional like-with-like: {'CONFIRMED' if func_sig else 'REFUTED'} (p={p_func:.2e})")
print(f"  Cross-modal composition link: {'CONFIRMED' if cross_sig else 'REFUTED'} (p={p_cross_perm:.4f})")

any_sig = trans_sig or func_sig or cross_sig
status = "CONFIRMED" if any_sig else "REFUTED"

results = {
    "hypothesis_id": "H32",
    "status": status,
    "transcriptomic_knn": {
        "subclass": {"mean_enrichment": round(float(mean_enrich_sub), 4), "frac_positive": round(float(frac_pos_sub), 4), "p": round(float(p_sub), 8)},
        "supertype": {"mean_enrichment": round(float(mean_enrich_st), 4), "frac_positive": round(float(frac_pos_st), 4), "p": round(float(p_st), 8)},
        "cluster": {"mean_enrichment": round(float(mean_enrich_cl), 4), "frac_positive": round(float(frac_pos_cl), 4), "p": round(float(p_cl), 8)},
        "glut_subclass": {"mean_enrichment": round(float(np.mean(diff_glut)), 4), "p": round(float(p_glut), 8)},
        "gaba_subclass": {"mean_enrichment": round(float(np.mean(diff_gaba)), 4), "p": round(float(p_gaba), 8)},
        "within_subclass_cluster": {"mean_enrichment": round(float(np.mean(diff_wsc)), 4), "frac_positive": round(float(np.mean(diff_wsc>0)), 4), "p": round(float(p_wsc), 8)},
    },
    "functional_spatial": {
        "same_cluster_mean_dist": round(float(np.mean(same_cluster_dists)), 2) if len(same_cluster_dists) > 0 else None,
        "diff_cluster_mean_dist": round(float(np.mean(diff_cluster_dists)), 2) if len(diff_cluster_dists) > 0 else None,
        "p": round(float(p_func), 8),
    },
    "cross_modal_composition": {
        "mantel_r": round(float(r_cross), 4),
        "mantel_p_perm": round(float(p_cross_perm), 6),
        "partial_r": round(float(r_cross_partial), 4),
        "partial_p": round(float(p_cross_partial), 6),
    },
    "connectivity_composition": {
        "conn_merfish_r": round(float(r_conn_mf), 4),
        "conn_merfish_p": round(float(p_conn_mf), 6),
        "conn_ephys_r": round(float(r_conn_ep), 4),
        "conn_ephys_p": round(float(p_conn_ep), 6),
    },
    "notes": f"Within-subclass cluster Δ={np.mean(diff_wsc):+.4f} p={p_wsc:.2e}; Func Δdist p={p_func:.2e}; Cross Mantel r={r_cross:+.4f} p={p_cross_perm:.4f}; {status}"
}

def json_safe(obj):
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [json_safe(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj

results = json_safe(results)

out_path = OUTPUT_DIR / "results.json"
out_path.write_text(json.dumps(results, indent=2))

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

ax = axes[0, 0]
levels = ["subclass", "supertype", "cluster"]
enrichments = [mean_enrich_sub, mean_enrich_st, mean_enrich_cl]
fracs_pos = [frac_pos_sub, frac_pos_st, frac_pos_cl]
x = np.arange(len(levels))
ax.bar(x - 0.15, enrichments, 0.3, label="Mean enrichment (Δ)", color="#4C72B0", alpha=0.8)
ax.bar(x + 0.15, fracs_pos, 0.3, label="Frac cells >0", color="#55A868", alpha=0.8)
ax.set_xticks(x)
ax.set_xticklabels(levels)
ax.axhline(0, color="black", linestyle="--", alpha=0.3)
ax.axhline(0.5, color="red", linestyle="--", alpha=0.3)
ax.set_title("MERFISH: transcriptomic like-with-like")
ax.legend(fontsize=8)

ax = axes[0, 1]
if len(same_cluster_dists) > 0:
    bins = np.linspace(0, max(np.percentile(same_cluster_dists, 95),
                               np.percentile(diff_cluster_dists, 95)), 50)
    ax.hist(same_cluster_dists, bins=bins, density=True, alpha=0.6, color="#4C72B0", label=f"Same (n={len(same_cluster_dists)})")
    ax.hist(diff_cluster_dists, bins=bins, density=True, alpha=0.6, color="#DD8452", label=f"Diff (n={len(diff_cluster_dists)})")
    ax.set_xlabel("Spatial distance (μm)")
    ax.set_ylabel("Density")
    ax.set_title(f"Functional: same vs diff cluster dist\np={p_func:.2e}")
    ax.legend(fontsize=8)

ax = axes[1, 0]
ax.scatter(d1, d2, s=80, alpha=0.8, c=["#4C72B0" if layer_vec[i] == layer_vec[j] else "#DD8452"
                                          for i in range(len(ALL_REGIONS))
                                          for j in range(i + 1, len(ALL_REGIONS))],
           edgecolors="black")
ax.set_xlabel("MERFISH composition distance")
ax.set_ylabel("Ephys composition distance")
ax.set_title(f"Cross-modal Mantel: r={r_cross:+.3f}, p={p_cross_perm:.3f}")
from matplotlib.patches import Patch
ax.legend(handles=[Patch(facecolor="#4C72B0", label="Within-layer"),
                   Patch(facecolor="#DD8452", label="Cross-layer")], fontsize=8)

ax = axes[1, 1]
ax.scatter(cd, d1, s=80, alpha=0.8, color="#55A868", edgecolors="black", label="MERFISH")
ax.scatter(cd, d2, s=80, alpha=0.8, color="#C44E52", edgecolors="black", label="Ephys")
ax.set_xlabel("Connectivity distance")
ax.set_ylabel("Composition distance")
ax.set_title(f"Connectivity ↔ composition\nMF r={r_conn_mf:+.3f}, EP r={r_conn_ep:+.3f}")
ax.legend(fontsize=8)

plt.suptitle("H32: Like-connects-with-like (cluster level)", fontsize=14)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "figures" / "like_connects_with_like.png", dpi=150, bbox_inches="tight")
plt.savefig(OUTPUT_DIR / "figures" / "like_connects_with_like.svg", bbox_inches="tight")

print(f"\nResults saved to {out_path}")
print("Done.")

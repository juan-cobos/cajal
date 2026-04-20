"""
Hypothesis H28: PFC subregion connectivity profiles (projection patterns
across brain targets) correlate with gene expression profiles across the
same subregions (Mantel test), controlling for layer.

If CONFIRMED: regions that project similarly express similar genes
(structural-molecular coupling).
If REFUTED: connectivity and transcriptomics are independent axes of
regional specialization.

Branch: connectivity-gene-mantel
Datasets: connectivity, merfish
"""

import json
import random
from itertools import combinations
from pathlib import Path

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr, pearsonr


def mantel_test(dist1, dist2, method="pearson", permutations=10000):
    d1 = dist1[np.triu_indices(dist1.shape[0], k=1)]
    d2 = dist2[np.triu_indices(dist2.shape[0], k=1)]
    if method == "pearson":
        r, _ = pearsonr(d1, d2)
    else:
        r, _ = spearmanr(d1, d2)
    if permutations <= 0:
        return float(r), 1.0
    perm_rs = []
    for _ in range(permutations):
        idx = np.random.permutation(len(d1))
        if method == "pearson":
            pr, _ = pearsonr(d1, d2[idx])
        else:
            pr, _ = spearmanr(d1, d2[idx])
        perm_rs.append(pr)
    p = (np.sum(np.abs(perm_rs) >= np.abs(r)) + 1) / (permutations + 1)
    return float(r), float(p)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

DATA_PATH = Path(__file__).parent.parent.parent / "data"
OUTPUT_DIR = Path(__file__).parent

L5_REGIONS = ["PL5", "ILA5", "ORBl5", "ORBm5", "ORBvl5"]
L6_REGIONS = ["PL6a", "ILA6a", "ORBl6a", "ORBm6a", "ORBvl6a"]
ALL_REGIONS = L5_REGIONS + L6_REGIONS

TARGET_GENES = {
    "IEG_adjacent": [
        "Fosl2", "Egr2", "Npas1", "Nr4a2", "Nr4a3", "Nrn1",
        "Cdkn1a", "Ppp1r17", "St18", "Rprm", "Ckap2l", "Mcm6",
        "Ccnd2", "Ccn3", "Ccn4",
    ],
    "calcium": [
        "Calb1", "Calb2", "Caln1", "Cacng3", "Trpc7", "Piezo2",
        "Pcp4l1", "Pvalb", "Nptx2", "Crym", "Calcb", "Calcr",
        "Creb3l1", "Car4",
    ],
    "neurotransmitter_signaling": [
        "Grm1", "Grm3", "Grm8", "Grin2c", "Grik1", "Grik3",
        "Drd1", "Drd2", "Drd3", "Drd5",
        "Htr2a", "Htr3a", "Htr1b", "Htr1d", "Htr7",
        "Nos1", "Crh", "Crhbp", "Penk", "Pdyn", "Tac2",
        "Gad2", "Slc17a7", "Slc32a1",
        "Adra1a", "Adra1b", "Cnr1", "Oprd1", "Oprk1",
        "Chrm2", "Chrm3",
    ],
    "ion_channels": [
        "Kcns3", "Kcnip1", "Kcnj5", "Kcnk9", "Kcnh8",
        "Kcng2", "Kcng1", "Kcnj8", "Kcnab3", "Kcnmb2",
        "Scn4b", "Scn5a", "Scn7a",
    ],
    "TF_identity": [
        "Bcl11b", "Rorb", "Nfib", "Nfix", "Foxo1",
        "Sox5", "Sox6", "Nr2f1", "Nr2f2", "Tcf7l2",
        "Pou3f1", "Pou3f3", "Emx2", "Otx2", "Eomes",
    ],
}

ALL_TARGETS = []
for group, genes in TARGET_GENES.items():
    ALL_TARGETS.extend(genes)
ALL_TARGETS = sorted(set(ALL_TARGETS))


def load_merfish_expression():
    adata = ad.read_h5ad(DATA_PATH / "merfish" / "C57BL6J-638850.h5ad")
    meta = pl.read_csv(DATA_PATH / "merfish" / "cell_metadata_with_parcellation_annotation.csv")
    pfc_cells = meta.filter(pl.col("parcellation_substructure").is_in(ALL_REGIONS))
    cell_labels = set(pfc_cells["cell_label"].to_list())
    adata = adata[adata.obs_names.isin(cell_labels)].copy()

    obs_df = pfc_cells.to_pandas().set_index("cell_label")
    common = adata.obs_names.intersection(obs_df.index)
    adata = adata[common].copy()
    adata.obs["parcellation_substructure"] = obs_df.loc[common, "parcellation_substructure"].values
    return adata


def build_connectivity_matrix():
    conn_matrix = {}
    for region in ALL_REGIONS:
        import glob
        files = glob.glob(str(DATA_PATH / "connectivity" / f"{region}_*.parquet"))
        if not files:
            print(f"  WARNING: No connectivity files for {region}")
            continue

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

    common_targets = set(conn_matrix[ALL_REGIONS[0]].keys())
    for region in ALL_REGIONS[1:]:
        if region in conn_matrix:
            common_targets &= set(conn_matrix[region].keys())
    common_targets = sorted(common_targets)

    mat = np.array([[conn_matrix[r].get(t, 0.0) for t in common_targets] for r in ALL_REGIONS])
    return mat, common_targets


print("=" * 60)
print("H28: Connectivity profiles ↔ Gene expression profiles")
print("=" * 60)

print("\n[1] Loading MERFISH expression...")
adata = load_merfish_expression()
print(f"  Loaded {adata.shape[0]} cells, {adata.shape[1]} genes")

gene_symbols = adata.var["gene_symbol"].to_dict()
available_genes = {v: k for k, v in gene_symbols.items()}
targets_present = [g for g in ALL_TARGETS if g in available_genes]
print(f"  Targets present: {len(targets_present)}/{len(ALL_TARGETS)}")

gene_expr = {}
for gene in targets_present:
    gene_idx = available_genes[gene]
    gene_expr[gene] = np.array(adata[:, gene_idx].X.todense()).flatten()

print("\n[2] Computing mean expression per subregion...")
subregions = adata.obs["parcellation_substructure"].values
expr_per_region = {}
for region in ALL_REGIONS:
    mask = subregions == region
    expr_per_region[region] = {}
    for gene in targets_present:
        expr_per_region[region][gene] = float(gene_expr[gene][mask].mean())

gene_matrix = np.array([[expr_per_region[r][g] for g in targets_present] for r in ALL_REGIONS])
print(f"  Gene matrix: {gene_matrix.shape}")

print("\n[3] Building connectivity matrix...")
conn_matrix, conn_targets = build_connectivity_matrix()
print(f"  Connectivity matrix: {conn_matrix.shape}")
print(f"  Common target structures: {len(conn_targets)}")

# Filter to targets with meaningful projections (NPV > 0.01 in at least 1 region)
active_mask = conn_matrix.max(axis=0) > 0.01
conn_filtered = conn_matrix[:, active_mask]
print(f"  Active targets (NPV>0.01 in ≥1 region): {conn_filtered.shape[1]}")

# Further filter: NPV > 0.1 in at least 3 regions
strong_mask = (conn_matrix > 0.1).sum(axis=0) >= 3
conn_strong = conn_matrix[:, strong_mask]
print(f"  Strong targets (NPV>0.1 in ≥3 regions): {conn_strong.shape[1]}")

print("\n[4] Computing distance matrices...")

# Gene expression distance (1 - Spearman correlation)
def corr_distance(matrix):
    n = matrix.shape[0]
    dist = np.zeros((n, n))
    for i, j in combinations(range(n), 2):
        r, _ = spearmanr(matrix[i], matrix[j])
        dist[i, j] = 1 - r
        dist[j, i] = 1 - r
    return dist

gene_dist = corr_distance(gene_matrix)
conn_dist_all = corr_distance(conn_matrix)
conn_dist_filtered = corr_distance(conn_filtered) if conn_filtered.shape[1] > 2 else conn_dist_all
conn_dist_strong = corr_distance(conn_strong) if conn_strong.shape[1] > 2 else conn_dist_all

print(f"\n  Gene distance matrix stats:")
gd_flat = gene_dist[np.triu_indices(len(ALL_REGIONS), k=1)]
print(f"    Mean: {gd_flat.mean():.4f}, Range: [{gd_flat.min():.4f}, {gd_flat.max():.4f}]")

print(f"\n  Connectivity distance matrix stats (all targets):")
cd_flat = conn_dist_all[np.triu_indices(len(ALL_REGIONS), k=1)]
print(f"    Mean: {cd_flat.mean():.4f}, Range: [{cd_flat.min():.4f}, {cd_flat.max():.4f}]")

print(f"\n  Connectivity distance matrix stats (strong targets):")
cds_flat = conn_dist_strong[np.triu_indices(len(ALL_REGIONS), k=1)]
print(f"    Mean: {cds_flat.mean():.4f}, Range: [{cds_flat.min():.4f}, {cds_flat.max():.4f}]")

print("\n[5] Mantel tests...")

# Mantel test: gene distance vs connectivity distance
def run_mantel(dist1, dist2, method="pearson", permutations=10000):
    return mantel_test(dist1, dist2, method=method, permutations=permutations)

# Full Mantel tests
print("  All targets:")
r_all, p_all = run_mantel(gene_dist, conn_dist_all)
print(f"    r={r_all:+.4f}, p={p_all:.4f}")

print("  Filtered targets (NPV>0.01):")
r_filt, p_filt = run_mantel(gene_dist, conn_dist_filtered)
print(f"    r={r_filt:+.4f}, p={p_filt:.4f}")

print("  Strong targets (NPV>0.1 in ≥3 regions):")
r_str, p_str = run_mantel(gene_dist, conn_dist_strong)
print(f"    r={r_str:+.4f}, p={p_str:.4f}")

print("\n[6] Within-layer Mantel tests...")

l5_idx = [ALL_REGIONS.index(r) for r in L5_REGIONS]
l6_idx = [ALL_REGIONS.index(r) for r in L6_REGIONS]

# L5 only
gene_dist_l5 = corr_distance(gene_matrix[l5_idx])
conn_dist_l5 = corr_distance(conn_strong[l5_idx])
r_l5, p_l5 = run_mantel(gene_dist_l5, conn_dist_l5, permutations=10000)
print(f"  L5 only (n={len(L5_REGIONS)}): r={r_l5:+.4f}, p={p_l5:.4f}")

# L6 only
gene_dist_l6 = corr_distance(gene_matrix[l6_idx])
conn_dist_l6 = corr_distance(conn_strong[l6_idx])
r_l6, p_l6 = run_mantel(gene_dist_l6, conn_dist_l6, permutations=10000)
print(f"  L6 only (n={len(L6_REGIONS)}): r={r_l6:+.4f}, p={p_l6:.4f}")

print("\n[7] Partial Mantel (controlling for layer distance)...")

layer_vec = np.array([0.0 if r in L5_REGIONS else 1.0 for r in ALL_REGIONS])
layer_dist = np.abs(layer_vec[:, None] - layer_vec[None, :])

# Partial Mantel: correlate gene_dist vs conn_dist, controlling for layer_dist
# Method: residualize gene_dist and conn_dist against layer_dist, then correlate residuals
gd_flat = gene_dist[np.triu_indices(len(ALL_REGIONS), k=1)]
cd_flat = conn_dist_strong[np.triu_indices(len(ALL_REGIONS), k=1)]
ld_flat = layer_dist[np.triu_indices(len(ALL_REGIONS), k=1)]

X = np.column_stack([np.ones(len(ld_flat)), ld_flat])
beta_gd = np.linalg.lstsq(X, gd_flat, rcond=None)[0]
gd_resid = gd_flat - X @ beta_gd

beta_cd = np.linalg.lstsq(X, cd_flat, rcond=None)[0]
cd_resid = cd_flat - X @ beta_cd

r_partial, p_partial = pearsonr(gd_resid, cd_resid)
print(f"  Partial Mantel (layer ctrl): r={r_partial:+.4f}, p={p_partial:.4f}")

# Permutation test for partial Mantel
n_perm = 10000
perm_rs_partial = []
for _ in range(n_perm):
    idx = np.random.permutation(len(gd_resid))
    pr, _ = pearsonr(gd_resid, cd_resid[idx])
    perm_rs_partial.append(pr)
p_partial_perm = (np.sum(np.abs(perm_rs_partial) >= np.abs(r_partial)) + 1) / (n_perm + 1)
print(f"  Partial Mantel permutation p: {p_partial_perm:.4f}")

print("\n[8] Cross-layer vs within-layer distance patterns...")

# Are L5-L5 pairs more similar than L5-L6 pairs?
within_layer_pairs = []
cross_layer_pairs = []
for i, j in combinations(range(len(ALL_REGIONS)), 2):
    li = 0 if ALL_REGIONS[i] in L5_REGIONS else 1
    lj = 0 if ALL_REGIONS[j] in L5_REGIONS else 1
    if li == lj:
        within_layer_pairs.append((i, j))
    else:
        cross_layer_pairs.append((i, j))

gd_within = np.array([gene_dist[i, j] for i, j in within_layer_pairs])
gd_cross = np.array([gene_dist[i, j] for i, j in cross_layer_pairs])
cd_within = np.array([conn_dist_strong[i, j] for i, j in within_layer_pairs])
cd_cross = np.array([conn_dist_strong[i, j] for i, j in cross_layer_pairs])

from scipy.stats import mannwhitneyu

print(f"  Gene distance: within-layer mean={gd_within.mean():.4f}, cross-layer mean={gd_cross.mean():.4f}")
u_gd, p_gd = mannwhitneyu(gd_within, gd_cross, alternative="less")
print(f"    Within < cross? U={u_gd:.1f}, p={p_gd:.4f}")

print(f"  Conn distance: within-layer mean={cd_within.mean():.4f}, cross-layer mean={cd_cross.mean():.4f}")
u_cd, p_cd = mannwhitneyu(cd_within, cd_cross, alternative="less")
print(f"    Within < cross? U={u_cd:.1f}, p={p_cd:.4f}")

print("\n[9] Layer-paired connectivity vs gene expression distance...")

# For each L5-L6 pair from same parent region, compute distance
parent_pairs = list(zip(L5_REGIONS, L6_REGIONS))
pair_gene_dist = []
pair_conn_dist = []
for l5r, l6r in parent_pairs:
    i = ALL_REGIONS.index(l5r)
    j = ALL_REGIONS.index(l6r)
    pair_gene_dist.append(gene_dist[i, j])
    pair_conn_dist.append(conn_dist_strong[i, j])

print(f"  L5↔L6 paired gene distances: {[f'{d:.4f}' for d in pair_gene_dist]}")
print(f"  L5↔L6 paired conn distances: {[f'{d:.4f}' for d in pair_conn_dist]}")

r_pair, p_pair = pearsonr(pair_gene_dist, pair_conn_dist)
print(f"  Paired distance correlation: r={r_pair:+.4f}, p={p_pair:.4f}")

status = "CONFIRMED" if p_str < 0.05 or p_partial_perm < 0.05 else "REFUTED"

results = {
    "hypothesis_id": "H28",
    "status": status,
    "n_regions": len(ALL_REGIONS),
    "n_conn_targets_all": conn_matrix.shape[1],
    "n_conn_targets_strong": conn_strong.shape[1],
    "n_genes": len(targets_present),
    "mantel_all": {"r": round(r_all, 4), "p": round(p_all, 6)},
    "mantel_filtered": {"r": round(r_filt, 4), "p": round(p_filt, 6)},
    "mantel_strong": {"r": round(r_str, 4), "p": round(p_str, 6)},
    "mantel_l5": {"r": round(r_l5, 4), "p": round(p_l5, 6)},
    "mantel_l6": {"r": round(r_l6, 4), "p": round(p_l6, 6)},
    "partial_mantel": {"r": round(r_partial, 4), "p_pearson": round(p_partial, 6), "p_perm": round(p_partial_perm, 6)},
    "layer_pair_distance": {"r": round(r_pair, 4), "p": round(p_pair, 6)},
    "within_vs_cross_layer": {
        "gene_dist": {"within_mean": round(float(gd_within.mean()), 4), "cross_mean": round(float(gd_cross.mean()), 4), "p": round(float(p_gd), 4)},
        "conn_dist": {"within_mean": round(float(cd_within.mean()), 4), "cross_mean": round(float(cd_cross.mean()), 4), "p": round(float(p_cd), 4)},
    },
    "notes": f"Mantel (strong targets) r={r_str:+.4f} p={p_str:.4f}; partial r={r_partial:+.4f} p={p_partial_perm:.4f}; {status}"
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
im = ax.imshow(gene_dist, cmap="viridis", vmin=0, vmax=1)
ax.set_xticks(range(len(ALL_REGIONS)))
ax.set_xticklabels(ALL_REGIONS, rotation=45, ha="right", fontsize=8)
ax.set_yticks(range(len(ALL_REGIONS)))
ax.set_yticklabels(ALL_REGIONS, fontsize=8)
ax.set_title("Gene expression distance")
plt.colorbar(im, ax=ax, label="1 - Spearman r")

ax = axes[0, 1]
im = ax.imshow(conn_dist_strong, cmap="viridis")
ax.set_xticks(range(len(ALL_REGIONS)))
ax.set_xticklabels(ALL_REGIONS, rotation=45, ha="right", fontsize=8)
ax.set_yticks(range(len(ALL_REGIONS)))
ax.set_yticklabels(ALL_REGIONS, fontsize=8)
ax.set_title("Connectivity distance (strong targets)")
plt.colorbar(im, ax=ax, label="1 - Spearman r")

ax = axes[1, 0]
ax.scatter(cd_flat, gd_flat, c=["#4C72B0" if layer_vec[i] == layer_vec[j] else "#DD8452"
                                 for i, j in combinations(range(len(ALL_REGIONS)), 2)],
           s=60, alpha=0.8, edgecolors="black")
ax.set_xlabel("Connectivity distance")
ax.set_ylabel("Gene expression distance")
ax.set_title(f"Mantel: r={r_str:+.3f}, p={p_str:.3f}")
from matplotlib.patches import Patch
ax.legend(handles=[Patch(facecolor="#4C72B0", label="Within-layer"),
                   Patch(facecolor="#DD8452", label="Cross-layer")], fontsize=8)

ax = axes[1, 1]
ax.scatter(cd_resid, gd_resid, s=60, alpha=0.8, color="#55A868", edgecolors="black")
ax.set_xlabel("Connectivity distance (layer residual)")
ax.set_ylabel("Gene expression distance (layer residual)")
ax.set_title(f"Partial Mantel: r={r_partial:+.3f}, p={p_partial_perm:.3f}")

plt.suptitle("H28: Connectivity ↔ Gene expression (Mantel test)", fontsize=14)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "figures" / "connectivity_gene_mantel.png", dpi=150, bbox_inches="tight")
plt.savefig(OUTPUT_DIR / "figures" / "connectivity_gene_mantel.svg", bbox_inches="tight")

print(f"\nResults saved to {out_path}")
print("Done.")

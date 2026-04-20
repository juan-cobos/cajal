"""
Hypothesis H27: Principal components of MERFISH gene co-expression across
PFC subregions predict burst index (and other ephys) after controlling for
layer, with fewer multiple comparisons than individual gene tests.

Rationale: H24/H25 tested 88 genes × 5 ephys properties = 440 tests.
PCA reduces 88 correlated genes to a few orthogonal axes. If PC1 captures
the layer-driven variation (Pou3f1, Rprm, etc.), PC2+ should capture
genuine gene→ephys signal with better power.

If CONFIRMED: gene co-expression modules predict ephys at pathway level.
If REFUTED: individual gene correlations don't cohere into modules.

Branch: gene-coexpression-pcs
Datasets: merfish, neural_activity
"""

import json
import random
from pathlib import Path

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from scipy.stats import spearmanr, pearsonr
from sklearn.decomposition import PCA

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
    adata.obs["layer"] = adata.obs["parcellation_substructure"].map(
        lambda x: "5" if x in L5_REGIONS else "6a"
    )
    adata.obs["neurotransmitter"] = obs_df.loc[common, "neurotransmitter"].values
    return adata


def partial_spearman(x, y, z):
    from scipy.stats import rankdata
    rx = rankdata(x)
    ry = rankdata(y)
    rz = rankdata(z)
    X = np.column_stack([np.ones(len(rx)), rz])
    beta_x = np.linalg.lstsq(X, rx, rcond=None)[0]
    res_x = rx - X @ beta_x
    beta_y = np.linalg.lstsq(X, ry, rcond=None)[0]
    res_y = ry - X @ beta_y
    r, p = pearsonr(res_x, res_y)
    return float(r), float(p)


print("=" * 60)
print("H27: Gene co-expression PCs → Electrophysiology")
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
expr_per_region_nt = {"Glut": {}, "GABA": {}}
for region in ALL_REGIONS:
    mask = subregions == region
    expr_per_region[region] = {}
    for gene in targets_present:
        expr_per_region[region][gene] = float(gene_expr[gene][mask].mean())

    for nt in ["Glut", "GABA"]:
        if region not in expr_per_region_nt[nt]:
            expr_per_region_nt[nt][region] = {}
        nt_mask = mask & (adata.obs["neurotransmitter"].values == nt)
        for gene in targets_present:
            expr_per_region_nt[nt][region][gene] = float(gene_expr[gene][nt_mask].mean())

print("\n[3] Loading neural activity...")
neural = pl.read_csv(DATA_PATH / "pfcmap" / "units.csv")
neural = neural.filter(pl.col("region").is_in(ALL_REGIONS))
neural = neural.filter(pl.col("utype").is_in(["ww", "nw"]))
neural = neural.filter(pl.col("rate_mean").is_not_null())

neural_props = {}
for region in ALL_REGIONS:
    sub = neural.filter(pl.col("region") == region)
    sub_m = sub.filter(pl.col("M_mean").is_not_null())
    ww = sub.filter(pl.col("utype") == "ww")
    nw = sub.filter(pl.col("utype") == "nw")
    neural_props[region] = {
        "rate_mean": float(sub["rate_mean"].mean()),
        "burst_mean": float(sub.filter(pl.col("B_mean").is_not_null())["B_mean"].mean()),
        "nw_frac": len(nw) / len(sub),
        "M_mean": float(sub_m["M_mean"].mean()) if len(sub_m) > 0 else None,
        "ww_burst": float(ww.filter(pl.col("B_mean").is_not_null())["B_mean"].mean()) if len(ww.filter(pl.col("B_mean").is_not_null())) > 0 else None,
        "n_total": len(sub),
    }

common_regions = sorted(set(expr_per_region.keys()) & set(neural_props.keys()))
layer_vec = np.array([0.0 if r in L5_REGIONS else 1.0 for r in common_regions])

print("\n[4] Building gene expression matrix (regions × genes)...")
X_all = np.array([[expr_per_region[r][g] for g in targets_present] for r in common_regions])
X_glut = np.array([[expr_per_region_nt["Glut"][r][g] for g in targets_present] for r in common_regions])
X_gaba = np.array([[expr_per_region_nt["GABA"][r][g] for g in targets_present] for r in common_regions])

print(f"  Matrix shape: {X_all.shape}")

print("\n[5] PCA on gene expression...")
pca_all = PCA(n_components=min(9, X_all.shape[1]), random_state=SEED)
X_all_z = (X_all - X_all.mean(axis=0)) / X_all.std(axis=0)
X_all_z = np.nan_to_num(X_all_z, nan=0.0)
pca_all.fit(X_all_z)

print(f"  Explained variance ratios:")
for i, vr in enumerate(pca_all.explained_variance_ratio_):
    print(f"    PC{i+1}: {vr:.3f} (cumulative: {pca_all.explained_variance_ratio_[:i+1].sum():.3f})")

loadings = pca_all.components_
print(f"\n  PC1 top loadings (absolute):")
pc1_order = np.argsort(np.abs(loadings[0]))[::-1]
for idx in pc1_order[:10]:
    g = targets_present[idx]
    for group, genes in TARGET_GENES.items():
        if g in genes:
            print(f"    {g:12s} ({group:20s}): {loadings[0, idx]:+.4f}")
            break

print(f"\n  PC2 top loadings (absolute):")
pc2_order = np.argsort(np.abs(loadings[1]))[::-1]
for idx in pc2_order[:10]:
    g = targets_present[idx]
    for group, genes in TARGET_GENES.items():
        if g in genes:
            print(f"    {g:12s} ({group:20s}): {loadings[1, idx]:+.4f}")
            break

pc_scores = pca_all.transform(X_all_z)

layer_pc_corrs = []
for i in range(min(9, X_all.shape[1])):
    r, p = pearsonr(pc_scores[:, i], layer_vec)
    layer_pc_corrs.append({"PC": i + 1, "r": round(float(r), 4), "p": round(float(p), 6), "R2": round(float(r**2), 4)})
    print(f"  Layer → PC{i+1}: r={r:+.4f}, p={p:.6f}, R²={r**2:.4f}")

pca_glut = PCA(n_components=min(9, X_glut.shape[1]), random_state=SEED)
X_glut_z = (X_glut - X_glut.mean(axis=0)) / X_glut.std(axis=0)
X_glut_z = np.nan_to_num(X_glut_z, nan=0.0)
pc_scores_glut = pca_glut.fit_transform(X_glut_z)

pca_gaba = PCA(n_components=min(9, X_gaba.shape[1]), random_state=SEED)
X_gaba_z = (X_gaba - X_gaba.mean(axis=0)) / X_gaba.std(axis=0)
X_gaba_z = np.nan_to_num(X_gaba_z, nan=0.0)
pc_scores_gaba = pca_gaba.fit_transform(X_gaba_z)

print("\n[6] PC scores → Electrophysiology correlations...")
ephys_props = ["burst_mean", "rate_mean", "nw_frac"]
ephys_vals = {}
for prop in ephys_props:
    ephys_vals[prop] = np.array([neural_props[r][prop] for r in common_regions])

n_pcs = min(9, X_all.shape[1])
pc_ephys_raw = {}
pc_ephys_partial = {}

for cell_type, scores in [("all", pc_scores), ("Glut", pc_scores_glut), ("GABA", pc_scores_gaba)]:
    for pc_i in range(scores.shape[1]):
        for prop in ephys_props:
            pv = ephys_vals[prop]
            if cell_type == "all":
                key = f"PC{pc_i+1}→{prop}"
            else:
                key = f"PC{pc_i+1}({cell_type})→{prop}"

            rho_raw, p_raw = spearmanr(scores[:, pc_i], pv)
            rho_partial, p_partial = partial_spearman(scores[:, pc_i], pv, layer_vec)
            pc_ephys_raw[key] = {"rho": round(float(rho_raw), 4), "p": round(float(p_raw), 6)}
            pc_ephys_partial[key] = {"rho": round(float(rho_partial), 4), "p": round(float(p_partial), 6)}

from statsmodels.stats.multitest import multipletests

raw_pvals = [v["p"] for v in pc_ephys_raw.values()]
partial_pvals = [v["p"] for v in pc_ephys_partial.values()]

reject_raw, pcorr_raw, _, _ = multipletests(raw_pvals, alpha=0.05, method="fdr_bh")
reject_partial, pcorr_partial, _, _ = multipletests(partial_pvals, alpha=0.05, method="fdr_bh")

n_raw_fdr = int(np.sum(reject_raw))
n_partial_fdr = int(np.sum(reject_partial))

print(f"\n  Total PC→ephys tests: {len(pc_ephys_raw)}")
print(f"  Raw FDR-sig: {n_raw_fdr}")
print(f"  Partial FDR-sig: {n_partial_fdr}")

print(f"\n  All raw PC→ephys correlations:")
for key, val in sorted(pc_ephys_raw.items(), key=lambda x: x[1]["p"]):
    idx = list(pc_ephys_raw.keys()).index(key)
    sig = "**" if reject_raw[idx] else "*" if val["p"] < 0.05 else ""
    partial_val = pc_ephys_partial[key]
    print(f"    {key:30s}: ρ_raw={val['rho']:+.4f}(p={val['p']:.4f}), ρ_partial={partial_val['rho']:+.4f}(p={partial_val['p']:.4f}){sig}")

print(f"\n  All partial PC→ephys correlations:")
for key, val in sorted(pc_ephys_partial.items(), key=lambda x: x[1]["p"]):
    idx = list(pc_ephys_partial.keys()).index(key)
    sig = "**" if reject_partial[idx] else "*" if val["p"] < 0.05 else ""
    print(f"    {key:30s}: ρ_partial={val['rho']:+.4f}, p={val['p']:.4f}{sig}")

print("\n[7] Comparison: PC-based vs individual gene-based prediction...")

top_gene_burst = 0
for gene in targets_present:
    gene_vals = np.array([expr_per_region[r][gene] for r in common_regions])
    burst_vals = ephys_vals["burst_mean"]
    valid = ~(np.isnan(gene_vals) | np.isnan(burst_vals))
    if valid.sum() < 4 or np.std(gene_vals[valid]) == 0:
        continue
    rho, p = spearmanr(gene_vals[valid], burst_vals[valid])
    if abs(rho) > top_gene_burst:
        top_gene_burst = abs(rho)

top_pc_burst = 0
for key, val in pc_ephys_raw.items():
    if "burst_mean" in key and "Glut" not in key and "GABA" not in key:
        if abs(val["rho"]) > top_pc_burst:
            top_pc_burst = abs(val["rho"])

print(f"  Top individual gene → burst |ρ|: {top_gene_burst:.4f}")
print(f"  Top PC → burst |ρ|: {top_pc_burst:.4f}")
print(f"  Individual gene tests: {440}")
print(f"  PC tests: {len(pc_ephys_raw)}")
print(f"  Multiple testing burden reduction: {440/len(pc_ephys_raw):.1f}x")

top_pc_burst_partial = 0
for key, val in pc_ephys_partial.items():
    if "burst_mean" in key and "Glut" not in key and "GABA" not in key:
        if abs(val["rho"]) > top_pc_burst_partial:
            top_pc_burst_partial = abs(val["rho"])

print(f"  Top PC → burst |ρ| (partial): {top_pc_burst_partial:.4f}")

status = "CONFIRMED" if n_partial_fdr > 0 else "REFUTED"

results = {
    "hypothesis_id": "H27",
    "status": status,
    "n_genes": len(targets_present),
    "n_pcs_all": pc_scores.shape[1],
    "n_pcs_glut": pc_scores_glut.shape[1],
    "n_pcs_gaba": pc_scores_gaba.shape[1],
    "n_total_pc_tests": len(pc_ephys_partial),
    "n_raw_fdr": n_raw_fdr,
    "n_partial_fdr": n_partial_fdr,
    "pca_explained_variance": {
        f"PC{i+1}": round(float(pca_all.explained_variance_ratio_[i]), 4)
        for i in range(n_pcs)
    },
    "layer_pc_correlations": layer_pc_corrs,
    "pc1_top_loadings": [
        {"gene": targets_present[idx], "loading": round(float(loadings[0, idx]), 4)}
        for idx in pc1_order[:10]
    ],
    "pc2_top_loadings": [
        {"gene": targets_present[idx], "loading": round(float(loadings[1, idx]), 4)}
        for idx in pc2_order[:10]
    ],
    "all_pc_ephys_partial": {
        k: v for k, v in sorted(pc_ephys_partial.items(), key=lambda x: x[1]["p"])
    },
    "fdr_partial_sig": [
        {"key": k, "rho": v["rho"], "p": v["p"], "p_corr": round(float(pcorr_partial[i]), 6)}
        for i, (k, v) in enumerate(pc_ephys_partial.items()) if reject_partial[i]
    ],
    "fdr_raw_sig": [
        {"key": k, "rho": v["rho"], "p": v["p"], "p_corr": round(float(pcorr_raw[i]), 6)}
        for i, (k, v) in enumerate(pc_ephys_raw.items()) if reject_raw[i]
    ],
    "comparison": {
        "top_gene_burst_rho": round(float(top_gene_burst), 4),
        "top_pc_burst_rho": round(float(top_pc_burst), 4),
        "top_pc_burst_rho_partial": round(float(top_pc_burst_partial), 4),
        "individual_gene_tests": 440,
        "pc_tests": len(pc_ephys_partial),
        "burden_reduction": round(float(440 / len(pc_ephys_partial)), 1),
    },
    "notes": f"{n_partial_fdr}/{len(pc_ephys_partial)} partial FDR-sig; {n_raw_fdr}/{len(pc_ephys_raw)} raw FDR-sig; {status}"
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

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

ax = axes[0, 0]
cumvar = np.cumsum(pca_all.explained_variance_ratio_)
ax.bar(range(1, n_pcs + 1), pca_all.explained_variance_ratio_, color="#4C72B0", alpha=0.8, label="Individual")
ax.step(range(1, n_pcs + 1), cumvar, where="mid", color="#C44E52", label="Cumulative")
ax.axhline(0.9, color="gray", linestyle="--", alpha=0.5)
ax.set_xlabel("Principal Component")
ax.set_ylabel("Explained Variance Ratio")
ax.set_title("PCA: Gene co-expression variance")
ax.legend()

ax = axes[0, 1]
for i in range(min(5, n_pcs)):
    load = loadings[i]
    ax.bar(range(len(targets_present)), load, alpha=0.5, label=f"PC{i+1}")
ax.set_xlabel("Gene index")
ax.set_ylabel("Loading")
ax.set_title("PCA loadings by gene")
ax.legend(fontsize=8)

gene_colors = []
for g in targets_present:
    for group, genes in TARGET_GENES.items():
        if g in genes:
            cmap = {"IEG_adjacent": "#DD8452", "calcium": "#55A868",
                     "neurotransmitter_signaling": "#4C72B0", "ion_channels": "#C44E52",
                     "TF_identity": "#8172B2"}
            gene_colors.append(cmap.get(group, "gray"))
            break
    else:
        gene_colors.append("gray")

ax.bar(range(len(targets_present)), loadings[0], color=gene_colors, alpha=0.8, edgecolor="none")
ax.set_xlabel("Gene index")
ax.set_ylabel("PC1 Loading")
ax.set_title("PC1 loadings (colored by gene group)")

ax = axes[1, 0]
burst_vals = ephys_vals["burst_mean"]
for i in range(min(5, n_pcs)):
    ax.scatter(pc_scores[:, i], burst_vals, s=80, alpha=0.7, label=f"PC{i+1}")
ax.set_xlabel("PC Score")
ax.set_ylabel("Burst Index")
ax.set_title("PC scores vs Burst Index")
ax.legend(fontsize=8)

ax = axes[1, 1]
pc_burst_partial_rhos = []
pc_burst_partial_ps = []
pc_labels = []
for key, val in sorted(pc_ephys_partial.items()):
    if "burst_mean" in key and "Glut" not in key and "GABA" not in key:
        pc_burst_partial_rhos.append(val["rho"])
        pc_burst_partial_ps.append(val["p"])
        pc_labels.append(key.replace("→burst_mean", ""))

if pc_burst_partial_rhos:
    y_pos = np.arange(len(pc_labels))
    ax.barh(y_pos, pc_burst_partial_rhos, color=["#4C72B0" if r > 0 else "#C44E52" for r in pc_burst_partial_rhos], alpha=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(pc_labels, fontsize=8)
    ax.axvline(0, color="black", linestyle="--", alpha=0.5)
    ax.set_xlabel("Partial Spearman ρ (layer ctrl)")
    ax.set_title("PC → Burst (partial, layer-controlled)")
    for i, p in enumerate(pc_burst_partial_ps):
        sig = "**" if p < 0.01 else "*" if p < 0.05 else ""
        ax.text(pc_burst_partial_rhos[i] + 0.02, i, f"p={p:.3f}{sig}", va="center", fontsize=7)

plt.suptitle("H27: Gene co-expression PCs → Electrophysiology", fontsize=14)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "figures" / "gene_coexpression_pcs.png", dpi=150, bbox_inches="tight")
plt.savefig(OUTPUT_DIR / "figures" / "gene_coexpression_pcs.svg", bbox_inches="tight")

print(f"\nResults saved to {out_path}")
print("Done.")

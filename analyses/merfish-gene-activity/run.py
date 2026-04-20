"""
Hypothesis H23: MERFISH expression of intrinsic excitability genes (K+
channels, glutamate receptors, neurotransmitter markers) per PFC subregion
correlates with electrophysiological firing properties (rate, burst index,
utype fraction) from neural_activity, within the same cortical layer.

Specifically: subregions with higher Slc17a7 (VGLUT1) / Grin2c / Kcns3
expression have higher WW firing rates, because these genes set excitability.

Branch: merfish-gene-activity
Datasets: merfish, neural_activity
"""

import json
import random
from pathlib import Path

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
from scipy.stats import spearmanr

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

DATA_PATH = Path(__file__).parent.parent.parent / "data"
OUTPUT_DIR = Path(__file__).parent

L5_REGIONS = ["PL5", "ILA5", "ORBl5", "ORBm5", "ORBvl5"]
L6_REGIONS = ["PL6a", "ILA6a", "ORBl6a", "ORBm6a", "ORBvl6a"]
ALL_REGIONS = L5_REGIONS + L6_REGIONS
PARENT_NAMES = ["PL", "ILA", "ORBl", "ORBm", "ORBvl"]

ACTIVITY_GENES = [
    "Slc17a7",
    "Slc32a1",
    "Gad2",
    "Grin2c",
    "Grik1",
    "Grik3",
    "Grm1",
    "Grm3",
    "Grm8",
    "Kcns3",
    "Kcnip1",
    "Kcnj5",
    "Kcnk9",
    "Kcnh8",
    "Kcng2",
    "Scn4b",
    "Scn5a",
    "Cacng3",
    "Drd1",
    "Drd2",
    "Htr2a",
    "Htr3a",
    "Nos1",
    "Pvalb",
    "Vip",
    "Crh",
    "Tac2",
    "Penk",
    "Pdyn",
    "Npas1",
    "Nr4a2",
    "Nr4a3",
    "Rgs4",
    "Calb1",
    "Calb2",
    "Rorb",
    "Bcl11b",
    "Satb2",
    "Reln",
]


def load_merfish_expression() -> ad.AnnData:
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
    return adata


def load_neural_activity() -> pl.DataFrame:
    df = pl.read_csv(DATA_PATH / "pfcmap" / "units.csv")
    df = df.filter(pl.col("region").is_in(ALL_REGIONS))
    df = df.filter(pl.col("utype").is_in(["ww", "nw"]))
    df = df.filter(pl.col("rate_mean").is_not_null())
    df = df.filter(pl.col("B_mean").is_not_null())
    return df


print("=" * 60)
print("H23: MERFISH Gene Expression → Neural Activity Properties")
print("=" * 60)

print("\n[1] Loading MERFISH expression...")
adata = load_merfish_expression()
print(f"  Loaded {adata.shape[0]} cells, {adata.shape[1]} genes")

gene_symbols = adata.var["gene_symbol"].to_dict()
available_genes = {v: k for k, v in gene_symbols.items()}

target_genes_present = [g for g in ACTIVITY_GENES if g in available_genes]
print(f"  Target genes present: {len(target_genes_present)}/{len(ACTIVITY_GENES)}")
print(f"  Missing: {set(ACTIVITY_GENES) - set(target_genes_present)}")

gene_expr = {}
for gene in target_genes_present:
    gene_idx = available_genes[gene]
    gene_expr[gene] = np.array(adata[:, gene_idx].X.todense()).flatten()

print("\n[2] Computing mean expression per subregion...")
subregions = adata.obs["parcellation_substructure"].values
expr_per_region = {}
for region in ALL_REGIONS:
    mask = subregions == region
    if mask.sum() == 0:
        continue
    expr_per_region[region] = {}
    for gene in target_genes_present:
        expr_per_region[region][gene] = float(gene_expr[gene][mask].mean())
    print(f"  {region}: n={mask.sum()}, VGLUT1={expr_per_region[region]['Slc17a7']:.3f}")

print("\n[3] Loading neural activity...")
neural = load_neural_activity()
print(f"  Loaded {len(neural)} neurons")

neural_props = {}
for region in ALL_REGIONS:
    sub = neural.filter(pl.col("region") == region)
    if len(sub) < 5:
        continue
    ww = sub.filter(pl.col("utype") == "ww")
    nw = sub.filter(pl.col("utype") == "nw")
    neural_props[region] = {
        "rate_mean": float(sub["rate_mean"].mean()),
        "burst_mean": float(sub["B_mean"].mean()),
        "nw_frac": len(nw) / len(sub),
        "ww_rate": float(ww["rate_mean"].mean()) if len(ww) > 0 else None,
        "nw_rate": float(nw["rate_mean"].mean()) if len(nw) > 0 else None,
        "ww_burst": float(ww["B_mean"].mean()) if len(ww) > 0 else None,
        "n_total": len(sub),
    }

common_regions = sorted(set(expr_per_region.keys()) & set(neural_props.keys()))
print(f"\n[4] Common regions: {len(common_regions)}")

print(f"\n{'='*60}")
print("L5-ONLY GENE vs ELECTROPHYSIOLOGY (n=5)")
print(f"{'='*60}")

l5_common = [r for r in common_regions if r in L5_REGIONS]
electro_props = ["rate_mean", "burst_mean", "nw_frac", "ww_rate", "ww_burst"]

gene_correlations = {}
for gene in target_genes_present:
    gene_vals = np.array([expr_per_region[r][gene] for r in l5_common])
    for prop in electro_props:
        prop_vals = np.array([neural_props[r][prop] for r in l5_common if neural_props[r][prop] is not None])
        if len(prop_vals) < len(l5_common):
            continue
        if np.std(gene_vals) == 0 or np.std(prop_vals) == 0:
            continue
        rho, p = spearmanr(gene_vals, prop_vals)
        key = f"{gene}→{prop}"
        gene_correlations[key] = {"gene": gene, "prop": prop, "rho": float(rho), "p": float(p)}

sorted_corrs = sorted(gene_correlations.items(), key=lambda x: x[1]["p"])
print(f"\nTop 20 correlations (L5, by p-value):")
for key, val in sorted_corrs[:20]:
    print(f"  {val['gene']:12s} → {val['prop']:12s}: ρ={val['rho']:+.3f}, p={val['p']:.3f}")

print(f"\n{'='*60}")
print("ALL-REGIONS GENE vs ELECTROPHYSIOLOGY (n=10)")
print(f"{'='*60}")

all_common = common_regions
gene_corrs_all = {}
for gene in target_genes_present:
    gene_vals = np.array([expr_per_region[r][gene] for r in all_common])
    for prop in ["rate_mean", "burst_mean", "nw_frac"]:
        prop_vals = np.array([neural_props[r][prop] for r in all_common])
        if np.std(gene_vals) == 0 or np.std(prop_vals) == 0:
            continue
        rho, p = spearmanr(gene_vals, prop_vals)
        key = f"{gene}→{prop}"
        gene_corrs_all[key] = {"gene": gene, "prop": prop, "rho": float(rho), "p": float(p)}

sorted_corrs_all = sorted(gene_corrs_all.items(), key=lambda x: x[1]["p"])
print(f"\nTop 20 correlations (all regions, by p-value):")
for key, val in sorted_corrs_all[:20]:
    print(f"  {val['gene']:12s} → {val['prop']:12s}: ρ={val['rho']:+.3f}, p={val['p']:.3f}")

n_tests_l5 = len(gene_correlations)
n_tests_all = len(gene_corrs_all)
print(f"\n  Total tests: L5={n_tests_l5}, All={n_tests_all}")

from statsmodels.stats.multitest import multipletests
l5_pvals = [v["p"] for v in gene_correlations.values()]
l5_reject, l5_pcorr, _, _ = multipletests(l5_pvals, alpha=0.05, method="fdr_bh")
n_sig_l5 = int(np.sum(l5_reject))

all_pvals = [v["p"] for v in gene_corrs_all.values()]
all_reject, all_pcorr, _, _ = multipletests(all_pvals, alpha=0.05, method="fdr_bh")
n_sig_all = int(np.sum(all_reject))

print(f"  FDR-significant: L5={n_sig_l5}/{n_tests_l5}, All={n_sig_all}/{n_tests_all}")

if n_sig_l5 > 0:
    print(f"\n  Significant L5 correlations:")
    for i, (key, val) in enumerate(gene_correlations.items()):
        if l5_reject[i]:
            print(f"    {val['gene']} → {val['prop']}: ρ={val['rho']:+.3f}, p_corr={l5_pcorr[i]:.4f}")

if n_sig_all > 0:
    print(f"\n  Significant All-region correlations:")
    for i, (key, val) in enumerate(gene_corrs_all.items()):
        if all_reject[i]:
            print(f"    {val['gene']} → {val['prop']}: ρ={val['rho']:+.3f}, p_corr={all_pcorr[i]:.4f}")

status = "CONFIRMED" if (n_sig_l5 > 0 or n_sig_all > 0) else "REFUTED"

top_genes_l5 = [(v["gene"], v["prop"], v["rho"], v["p"]) for _, v in sorted_corrs_l5[:5]] if n_sig_l5 == 0 else [(v["gene"], v["prop"], v["rho"], v["p"]) for i, (k, v) in enumerate(gene_correlations.items()) if l5_reject[i]]

results = {
    "hypothesis_id": "H23",
    "status": status,
    "n_genes_tested": len(target_genes_present),
    "n_tests_l5": n_tests_l5,
    "n_tests_all": n_tests_all,
    "n_fdr_sig_l5": n_sig_l5,
    "n_fdr_sig_all": n_sig_all,
    "top_l5_unadjusted": [{"gene": v["gene"], "prop": v["prop"], "rho": round(v["rho"], 4), "p": round(v["p"], 4)} for _, v in sorted_corrs[:10]],
    "top_all_unadjusted": [{"gene": v["gene"], "prop": v["prop"], "rho": round(v["rho"], 4), "p": round(v["p"], 4)} for _, v in sorted_corrs_all[:10]],
    "notes": f"L5 FDR-sig: {n_sig_l5}; All FDR-sig: {n_sig_all}; {status}"
}

print("\n" + json.dumps(results, indent=2))

out_path = OUTPUT_DIR / "results.json"
out_path.write_text(json.dumps(results, indent=2))

# Visualization: heatmap of gene-eprop correlations
l5_genes = target_genes_present
l5_props = ["rate_mean", "burst_mean", "nw_frac"]
l5_corr_matrix = np.zeros((len(l5_genes), len(l5_props)))
l5_pval_matrix = np.ones((len(l5_genes), len(l5_props)))

for i, gene in enumerate(l5_genes):
    gene_vals = np.array([expr_per_region[r][gene] for r in l5_common])
    for j, prop in enumerate(l5_props):
        prop_vals = np.array([neural_props[r][prop] for r in l5_common])
        if np.std(gene_vals) > 0 and np.std(prop_vals) > 0:
            rho, p = spearmanr(gene_vals, prop_vals)
            l5_corr_matrix[i, j] = rho
            l5_pval_matrix[i, j] = p

fig, axes = plt.subplots(1, 2, figsize=(14, 16))

im = axes[0].imshow(l5_corr_matrix, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
axes[0].set_xticks(range(len(l5_props)))
axes[0].set_xticklabels(l5_props, rotation=45, ha="right")
axes[0].set_yticks(range(len(l5_genes)))
axes[0].set_yticklabels(l5_genes, fontsize=8)
axes[0].set_title("L5: Gene vs Ephys (Spearman ρ)")
for i in range(len(l5_genes)):
    for j in range(len(l5_props)):
        val = l5_corr_matrix[i, j]
        pval = l5_pval_matrix[i, j]
        sig = "*" if pval < 0.05 else ""
        axes[0].text(j, i, f"{val:.2f}{sig}", ha="center", va="center", fontsize=6,
                     color="white" if abs(val) > 0.6 else "black")
plt.colorbar(im, ax=axes[0], shrink=0.5)

# All-regions heatmap
all_corr = np.zeros((len(l5_genes), len(l5_props)))
all_pval = np.ones((len(l5_genes), len(l5_props)))
for i, gene in enumerate(l5_genes):
    gene_vals = np.array([expr_per_region[r][gene] for r in all_common])
    for j, prop in enumerate(l5_props):
        prop_vals = np.array([neural_props[r][prop] for r in all_common])
        if np.std(gene_vals) > 0 and np.std(prop_vals) > 0:
            rho, p = spearmanr(gene_vals, prop_vals)
            all_corr[i, j] = rho
            all_pval[i, j] = p

im2 = axes[1].imshow(all_corr, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
axes[1].set_xticks(range(len(l5_props)))
axes[1].set_xticklabels(l5_props, rotation=45, ha="right")
axes[1].set_yticks(range(len(l5_genes)))
axes[1].set_yticklabels(l5_genes, fontsize=8)
axes[1].set_title("All Regions: Gene vs Ephys (Spearman ρ)")
for i in range(len(l5_genes)):
    for j in range(len(l5_props)):
        val = all_corr[i, j]
        pval = all_pval[i, j]
        sig = "*" if pval < 0.05 else ""
        axes[1].text(j, i, f"{val:.2f}{sig}", ha="center", va="center", fontsize=6,
                     color="white" if abs(val) > 0.6 else "black")
plt.colorbar(im2, ax=axes[1], shrink=0.5)

plt.suptitle("H23: MERFISH Gene Expression → Neural Activity", fontsize=14)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "figures" / "merfish_gene_activity.png", dpi=150, bbox_inches="tight")
plt.savefig(OUTPUT_DIR / "figures" / "merfish_gene_activity.svg", bbox_inches="tight")

print(f"\nResults saved to {out_path}")

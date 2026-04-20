"""
Hypothesis H24: MERFISH expression of IEG-adjacent (Fosl2, Egr2, Npas1,
Nr4a2, Nr4a3, Nrn1, Cdkn1a) and calcium-related (Calb1, Calb2, Cacng3,
Trpc7, Piezo2, Pcp4l1, Pvalb, Nptx2, Crym, Caln1) genes per PFC subregion
correlates with burst index — the most gene-tractable electrophysiological
feature identified in H23.

Branch: ieg-calcium-activity
Datasets: merfish, neural_activity
"""

import json
import random
from pathlib import Path

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
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
    adata.obs["neurotransmitter"] = obs_df.loc[common, "neurotransmitter"].values
    adata.obs["subclass"] = obs_df.loc[common, "subclass"].values
    return adata


def load_neural_activity() -> pl.DataFrame:
    df = pl.read_csv(DATA_PATH / "pfcmap" / "units.csv")
    df = df.filter(pl.col("region").is_in(ALL_REGIONS))
    df = df.filter(pl.col("utype").is_in(["ww", "nw"]))
    df = df.filter(pl.col("rate_mean").is_not_null())
    df = df.filter(pl.col("B_mean").is_not_null())
    return df


print("=" * 60)
print("H24: IEG + Calcium Genes → Burst Index & Other Ephys")
print("=" * 60)

print("\n[1] Loading MERFISH expression...")
adata = load_merfish_expression()
print(f"  Loaded {adata.shape[0]} cells, {adata.shape[1]} genes")

gene_symbols = adata.var["gene_symbol"].to_dict()
available_genes = {v: k for k, v in gene_symbols.items()}
targets_present = [g for g in ALL_TARGETS if g in available_genes]
targets_missing = [g for g in ALL_TARGETS if g not in available_genes]
print(f"  Targets present: {len(targets_present)}/{len(ALL_TARGETS)}")
print(f"  Missing: {targets_missing}")

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

    glut_mask = mask & (adata.obs["neurotransmitter"].values == "Glut")
    gaba_mask = mask & (adata.obs["neurotransmitter"].values == "GABA")
    expr_per_region[region]["_n_glut"] = int(glut_mask.sum())
    expr_per_region[region]["_n_gaba"] = int(gaba_mask.sum())
    print(f"  {region}: n={mask.sum()}, Glut={glut_mask.sum()}, GABA={gaba_mask.sum()}")

print("\n[3] Also computing Glut-only and GABA-only expression...")
expr_per_region_nt = {}
for region in ALL_REGIONS:
    expr_per_region_nt[region] = {"Glut": {}, "GABA": {}}
    rmask = subregions == region
    for nt in ["Glut", "GABA"]:
        nt_mask = rmask & (adata.obs["neurotransmitter"].values == nt)
        for gene in targets_present:
            gene_idx = available_genes[gene]
            expr_per_region_nt[region][nt][gene] = float(
                np.array(adata[nt_mask, gene_idx].X.todense()).flatten().mean()
            )

print("\n[4] Loading neural activity...")
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
        "ww_burst": float(ww["B_mean"].mean()) if len(ww) > 0 else None,
        "n_total": len(sub),
    }

common_regions = sorted(set(expr_per_region.keys()) & set(neural_props.keys()))

electro_props = ["rate_mean", "burst_mean", "nw_frac"]
electro_labels = {"rate_mean": "Mean firing rate", "burst_mean": "Burst index", "nw_frac": "NW fraction"}

print(f"\n[5] Correlating genes vs ephys across {len(common_regions)} regions...")

gene_corrs = {}
for gene in targets_present:
    for prop in electro_props:
        gene_vals = np.array([expr_per_region[r][gene] for r in common_regions])
        prop_vals = np.array([neural_props[r][prop] for r in common_regions])
        if np.std(gene_vals) == 0 or np.std(prop_vals) == 0:
            continue
        rho, p = spearmanr(gene_vals, prop_vals)
        gene_corrs[f"{gene}→{prop}"] = {
            "gene": gene, "prop": prop,
            "rho": float(rho), "p": float(p),
            "n": len(common_regions),
        }

for gene in targets_present:
    for prop in ["burst_mean"]:
        for nt in ["Glut", "GABA"]:
            gene_vals = np.array([expr_per_region_nt[r][nt][gene] for r in common_regions])
            prop_vals = np.array([neural_props[r][prop] for r in common_regions])
            if np.std(gene_vals) == 0 or np.std(prop_vals) == 0:
                continue
            rho, p = spearmanr(gene_vals, prop_vals)
            gene_corrs[f"{gene}→{prop}_{nt}"] = {
                "gene": gene, "prop": f"{prop}_{nt}",
                "rho": float(rho), "p": float(p),
                "n": len(common_regions),
            }

from statsmodels.stats.multitest import multipletests
all_pvals = [v["p"] for v in gene_corrs.values()]
reject, pcorr, _, _ = multipletests(all_pvals, alpha=0.05, method="fdr_bh")
n_sig = int(np.sum(reject))

sorted_corrs = sorted(gene_corrs.items(), key=lambda x: x[1]["p"])

print(f"\nTotal tests: {len(gene_corrs)}")
print(f"FDR-significant: {n_sig}")

print(f"\nTop 30 unadjusted:")
for key, val in sorted_corrs[:30]:
    sig = "*" if reject[list(gene_corrs.keys()).index(key)] else ""
    print(f"  {val['gene']:12s} → {val['prop']:20s}: ρ={val['rho']:+.3f}, p={val['p']:.4f}{sig}")

if n_sig > 0:
    print(f"\nAll FDR-significant ({n_sig}):")
    for i, (key, val) in enumerate(gene_corrs.items()):
        if reject[i]:
            group = "UNKNOWN"
            for g, genes in TARGET_GENES.items():
                if val["gene"] in genes:
                    group = g
                    break
            print(f"  {val['gene']:12s} ({group:20s}) → {val['prop']:20s}: ρ={val['rho']:+.3f}, p_corr={pcorr[i]:.4f}")

print(f"\n[6] Group-level summary:")
for group, genes in TARGET_GENES.items():
    group_genes = [g for g in genes if g in available_genes]
    group_sig = sum(1 for i, (k, v) in enumerate(gene_corrs.items())
                    if v["gene"] in group_genes and reject[i])
    group_total = sum(1 for k, v in gene_corrs.items() if v["gene"] in group_genes)
    group_top = min(3, len(group_genes))
    top_genes_in_group = sorted(
        [(v["gene"], v["prop"], v["rho"], v["p"])
         for k, v in gene_corrs.items() if v["gene"] in group_genes],
        key=lambda x: x[3]
    )[:group_top]
    print(f"  {group:30s}: {group_sig}/{group_total} FDR-sig; top: {[(g,p,f'{r:+.2f}') for g,p,r,_ in top_genes_in_group]}")

status = "CONFIRMED" if n_sig > 0 else "REFUTED"

results = {
    "hypothesis_id": "H24",
    "status": status,
    "n_genes_tested": len(targets_present),
    "n_tests": len(gene_corrs),
    "n_fdr_sig": n_sig,
    "gene_groups": {g: len(genes) for g, genes in TARGET_GENES.items()},
    "top_unadjusted": [
        {"gene": v["gene"], "prop": v["prop"], "rho": round(v["rho"], 4), "p": round(v["p"], 6)}
        for _, v in sorted_corrs[:20]
    ],
    "fdr_sig_list": [
        {"gene": v["gene"], "prop": v["prop"], "rho": round(v["rho"], 4), "p_corr": round(pcorr[i], 6)}
        for i, (k, v) in enumerate(gene_corrs.items()) if reject[i]
    ],
    "notes": f"{n_sig}/{len(gene_corrs)} FDR-sig; {status}"
}

print("\n" + json.dumps(results, indent=2))

out_path = OUTPUT_DIR / "results.json"
out_path.write_text(json.dumps(results, indent=2))

fig, axes = plt.subplots(1, 3, figsize=(18, 10))

for ax_idx, prop in enumerate(["burst_mean", "rate_mean", "nw_frac"]):
    prop_genes = [(v["gene"], v["rho"], v["p"]) for k, v in gene_corrs.items()
                  if v["prop"] == prop and not v["prop"].endswith("_Glut") and not v["prop"].endswith("_GABA")]
    prop_genes.sort(key=lambda x: x[2])
    top_n = min(15, len(prop_genes))
    top_genes = prop_genes[:top_n]
    names = [g for g, _, _ in top_genes]
    rhos = [r for _, r, _ in top_genes]
    pvals = [p for _, _, p in top_genes]

    colors = []
    for g in names:
        for group, genes in TARGET_GENES.items():
            if g in genes:
                cmap = {"IEG_adjacent": "#DD8452", "calcium": "#55A868",
                         "neurotransmitter_signaling": "#4C72B0", "ion_channels": "#C44E52",
                         "TF_identity": "#8172B2"}
                colors.append(cmap.get(group, "gray"))
                break
        else:
            colors.append("gray")

    ax = axes[ax_idx]
    y_pos = np.arange(len(names))
    ax.barh(y_pos, rhos, color=colors, edgecolor="black", alpha=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("Spearman ρ")
    ax.axvline(0, color="black", linestyle="--", alpha=0.5)
    label = electro_labels.get(prop, prop)
    ax.set_title(f"Top genes → {label}\n(n={len(common_regions)})")

    for i, (g, r, p) in enumerate(top_genes):
        sig = "**" if p < 0.01 else "*" if p < 0.05 else ""
        ax.text(r + 0.02 if r > 0 else r - 0.02, i, f"p={p:.3f}{sig}",
                va="center", ha="left" if r > 0 else "right", fontsize=7)

from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor="#DD8452", edgecolor="black", label="IEG-adjacent"),
    Patch(facecolor="#55A868", edgecolor="black", label="Calcium"),
    Patch(facecolor="#4C72B0", edgecolor="black", label="Neurotransmitter"),
    Patch(facecolor="#C44E52", edgecolor="black", label="Ion channels"),
    Patch(facecolor="#8172B2", edgecolor="black", label="TF/Identity"),
]
axes[2].legend(handles=legend_elements, loc="lower right", fontsize=8)

plt.suptitle("H24: IEG + Calcium Genes → Electrophysiology", fontsize=14)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "figures" / "ieg_calcium_activity.png", dpi=150, bbox_inches="tight")
plt.savefig(OUTPUT_DIR / "figures" / "ieg_calcium_activity.svg", bbox_inches="tight")

print(f"\nResults saved to {out_path}")

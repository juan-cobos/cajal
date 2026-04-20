"""
Hypothesis H25: The FDR-significant gene→burst correlations from H24 remain
significant after controlling for layer (L5 vs L6) via partial correlation.

If CONFIRMED: gene→ephys links are genuine, independent of laminar confound.
If REFUTED: H24 findings are driven by the L5/L6 dichotomy.

Branch: layer-controlled-gene-burst
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

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

DATA_PATH = Path(__file__).parent.parent.parent / "data"
OUTPUT_DIR = Path(__file__).parent

L5_REGIONS = ["PL5", "ILA5", "ORBl5", "ORBm5", "ORBvl5"]
L6_REGIONS = ["PL6a", "ILA6a", "ORBl6a", "ORBm6a", "ORBvl6a"]
ALL_REGIONS = L5_REGIONS + L6_REGIONS

H24_FDR_SIG = [
    {"gene": "Grm1", "prop": "burst_mean", "rho_orig": -0.8667},
    {"gene": "Mcm6", "prop": "burst_mean", "rho_orig": -0.9273},
    {"gene": "Nos1", "prop": "burst_mean", "rho_orig": -0.8788},
    {"gene": "Piezo2", "prop": "burst_mean", "rho_orig": -0.9152},
    {"gene": "Pvalb", "prop": "burst_mean", "rho_orig": 0.8667},
    {"gene": "St18", "prop": "nw_frac", "rho_orig": 0.9273},
    {"gene": "Gad2", "prop": "burst_mean_GABA", "rho_orig": -0.8667},
    {"gene": "Kcnab3", "prop": "burst_mean_Glut", "rho_orig": 0.8909},
    {"gene": "Mcm6", "prop": "burst_mean_Glut", "rho_orig": -0.9273},
    {"gene": "Pvalb", "prop": "burst_mean_Glut", "rho_orig": 0.8909},
    {"gene": "Scn4b", "prop": "burst_mean_Glut", "rho_orig": 0.8667},
]

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


def partial_spearman(x, y, z):
    """Partial Spearman correlation: correlate x,y after removing effect of z.

    Uses rank transformation + residualization approach.
    Returns (rho_partial, p_value).
    """
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
    df = len(x) - 3
    return float(r), float(p), df


print("=" * 60)
print("H25: Layer-controlled gene→burst correlations")
print("=" * 60)

print("\n[1] Loading MERFISH expression...")
adata = load_merfish_expression()
print(f"  Loaded {adata.shape[0]} cells, {adata.shape[1]} genes")

gene_symbols = adata.var["gene_symbol"].to_dict()
available_genes = {v: k for k, v in gene_symbols.items()}
targets_present = [g for g in ALL_TARGETS if g in available_genes]

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
        "ww_burst": float(ww["B_mean"].mean()) if len(ww) > 0 else None,
        "n_total": len(sub),
    }

common_regions = sorted(set(expr_per_region.keys()) & set(neural_props.keys()))
print(f"  Common regions: {len(common_regions)}")

layer_vec = np.array([0.0 if r in L5_REGIONS else 1.0 for r in common_regions])

print("\n[4] Computing layer-controlled partial correlations for H24 FDR-sig pairs...")

def get_gene_vals(gene, regions, nt=None):
    if nt:
        return np.array([expr_per_region_nt[r][nt][gene] for r in regions])
    return np.array([expr_per_region[r][gene] for r in regions])


def get_prop_vals(prop, regions):
    vals = []
    for r in regions:
        if prop == "burst_mean_Glut":
            sub = neural.filter(pl.col("region") == r)
            ww = sub.filter((pl.col("utype") == "ww"))
            vals.append(float(ww["B_mean"].mean()) if len(ww) > 0 else np.nan)
        elif prop == "burst_mean_GABA":
            sub = neural.filter(pl.col("region") == r)
            nw = sub.filter((pl.col("utype") == "nw"))
            vals.append(float(nw["B_mean"].mean()) if len(nw) > 0 else np.nan)
        else:
            vals.append(neural_props[r][prop])
    return np.array(vals)


partial_results = []
for entry in H24_FDR_SIG:
    gene = entry["gene"]
    prop = entry["prop"]
    rho_orig = entry["rho_orig"]

    nt = None
    prop_clean = prop
    if prop.endswith("_Glut"):
        nt = "Glut"
        prop_clean = prop.replace("_Glut", "")
    elif prop.endswith("_GABA"):
        nt = "GABA"
        prop_clean = prop.replace("_GABA", "")

    gene_vals = get_gene_vals(gene, common_regions, nt=nt)
    prop_vals = get_prop_vals(prop, common_regions)

    valid = ~(np.isnan(gene_vals) | np.isnan(prop_vals))
    gv = gene_vals[valid]
    pv = prop_vals[valid]
    lv = layer_vec[valid]

    if len(gv) < 4:
        continue

    rho_raw, p_raw = spearmanr(gv, pv)
    rho_partial, p_partial, df = partial_spearman(gv, pv, lv)

    l5_mask = lv == 0
    l6_mask = lv == 1
    rho_l5 = rho_l5_p = None
    rho_l6 = rho_l6_p = None
    if l5_mask.sum() >= 3:
        rho_l5, rho_l5_p = spearmanr(gv[l5_mask], pv[l5_mask])
    if l6_mask.sum() >= 3:
        rho_l6, rho_l6_p = spearmanr(gv[l6_mask], pv[l6_mask])

    direction_consistent = None
    if rho_l5 is not None and rho_l6 is not None:
        direction_consistent = (np.sign(rho_l5) == np.sign(rho_l6)) and (np.sign(rho_l5) == np.sign(rho_raw))

    partial_results.append({
        "gene": gene,
        "prop": prop,
        "rho_raw": round(rho_raw, 4),
        "p_raw": round(p_raw, 6),
        "rho_partial": round(rho_partial, 4),
        "p_partial": round(p_partial, 6),
        "df": df,
        "rho_l5": round(rho_l5, 4) if rho_l5 is not None else None,
        "p_l5": round(rho_l5_p, 6) if rho_l5_p is not None else None,
        "rho_l6": round(rho_l6, 4) if rho_l6 is not None else None,
        "p_l6": round(rho_l6_p, 6) if rho_l6_p is not None else None,
        "direction_consistent": direction_consistent,
        "variance_removed": round(1 - (rho_partial / rho_raw) ** 2 if rho_raw != 0 else 0, 4) if abs(rho_raw) > abs(rho_partial) else 0,
    })

    print(f"\n  {gene} → {prop}:")
    print(f"    Raw:     ρ={rho_raw:+.4f}, p={p_raw:.6f}")
    print(f"    Partial: ρ={rho_partial:+.4f}, p={p_partial:.6f} (df={df})")
    print(f"    L5 only: ρ={rho_l5:+.4f}, p={rho_l5_p:.4f}" if rho_l5 is not None else "    L5 only: n/a")
    print(f"    L6 only: ρ={rho_l6:+.4f}, p={rho_l6_p:.4f}" if rho_l6 is not None else "    L6 only: n/a")
    if direction_consistent is not None:
        print(f"    Direction consistent across layers: {direction_consistent}")

print("\n\n[5] Full gene screen with partial correlation...")

all_corrs_raw = {}
all_corrs_partial = {}

for gene in targets_present:
    for prop in ["rate_mean", "burst_mean", "nw_frac"]:
        gene_vals = get_gene_vals(gene, common_regions)
        prop_vals = get_prop_vals(prop, common_regions)
        valid = ~(np.isnan(gene_vals) | np.isnan(prop_vals))
        gv, pv, lv = gene_vals[valid], prop_vals[valid], layer_vec[valid]
        if len(gv) < 4 or np.std(gv) == 0 or np.std(pv) == 0:
            continue
        rho_raw, p_raw = spearmanr(gv, pv)
        rho_partial, p_partial, df = partial_spearman(gv, pv, lv)
        key = f"{gene}→{prop}"
        all_corrs_raw[key] = {"rho": rho_raw, "p": p_raw}
        all_corrs_partial[key] = {"rho": rho_partial, "p": p_partial, "df": df}

    for prop_base in ["burst_mean"]:
        for nt in ["Glut", "GABA"]:
            gene_vals = get_gene_vals(gene, common_regions, nt=nt)
            prop_vals = get_prop_vals(f"{prop_base}_{nt}", common_regions)
            valid = ~(np.isnan(gene_vals) | np.isnan(prop_vals))
            gv, pv, lv = gene_vals[valid], prop_vals[valid], layer_vec[valid]
            if len(gv) < 4 or np.std(gv) == 0 or np.std(pv) == 0:
                continue
            rho_raw, p_raw = spearmanr(gv, pv)
            rho_partial, p_partial, df = partial_spearman(gv, pv, lv)
            key = f"{gene}→{prop_base}_{nt}"
            all_corrs_raw[key] = {"rho": rho_raw, "p": p_raw}
            all_corrs_partial[key] = {"rho": rho_partial, "p": p_partial, "df": df}

from statsmodels.stats.multitest import multipletests

partial_pvals = [v["p"] for v in all_corrs_partial.values()]
reject_partial, pcorr_partial, _, _ = multipletests(partial_pvals, alpha=0.05, method="fdr_bh")
n_partial_fdr = int(np.sum(reject_partial))

raw_pvals = [v["p"] for v in all_corrs_raw.values()]
reject_raw, pcorr_raw, _, _ = multipletests(raw_pvals, alpha=0.05, method="fdr_bh")
n_raw_fdr = int(np.sum(reject_raw))

print(f"\n  Raw correlations FDR-sig: {n_raw_fdr}/{len(all_corrs_raw)}")
print(f"  Partial correlations FDR-sig: {n_partial_fdr}/{len(all_corrs_partial)}")

print(f"\n  Top 15 partial correlations:")
sorted_partial = sorted(all_corrs_partial.items(), key=lambda x: x[1]["p"])
for key, val in sorted_partial[:15]:
    idx = list(all_corrs_partial.keys()).index(key)
    sig = "**" if reject_partial[idx] else "*" if val["p"] < 0.05 else ""
    raw_rho = all_corrs_raw[key]["rho"]
    delta = val["rho"] - raw_rho
    print(f"    {key:40s}: ρ_raw={raw_rho:+.3f} → ρ_partial={val['rho']:+.3f} (Δ={delta:+.3f}), p={val['p']:.4f}{sig}")

print("\n\n[6] How much variance does layer explain?")

from scipy.stats import pointbiserialr
for prop in ["burst_mean", "rate_mean", "nw_frac"]:
    prop_vals = get_prop_vals(prop, common_regions)
    r, p = pointbiserialr(layer_vec, prop_vals)
    print(f"  Layer → {prop}: r={r:+.3f}, p={p:.4f}, R²={r**2:.3f}")

layer_expr_corrs = {}
for gene in targets_present:
    gene_vals = get_gene_vals(gene, common_regions)
    r, p = pointbiserialr(layer_vec, gene_vals)
    layer_expr_corrs[gene] = {"r": round(r, 4), "p": round(p, 6), "R2": round(r**2, 4)}

print(f"\n  Top 10 genes most associated with layer:")
sorted_layer = sorted(layer_expr_corrs.items(), key=lambda x: abs(x[1]["r"]), reverse=True)
for gene, val in sorted_layer[:10]:
    print(f"    {gene:12s}: r={val['r']:+.4f}, p={val['p']:.6f}, R²={val['R2']:.4f}")

n_survive = sum(1 for r in partial_results if r["p_partial"] < 0.05)
n_direction = sum(1 for r in partial_results if r["direction_consistent"] == True)

print(f"\n\n[7] Summary for H24 FDR-sig pairs:")
print(f"  Survive p<0.05 after layer control: {n_survive}/11")
print(f"  Direction consistent across L5 & L6: {n_direction}/11")

status = "CONFIRMED" if n_survive >= 3 else "REFUTED"
print(f"\n  Hypothesis status: {status}")

results = {
    "hypothesis_id": "H25",
    "status": status,
    "n_h24_fdr_pairs": 11,
    "n_survive_partial_p005": n_survive,
    "n_direction_consistent": n_direction,
    "partial_results": partial_results,
    "n_raw_fdr": n_raw_fdr,
    "n_partial_fdr": n_partial_fdr,
    "n_total_tests": len(all_corrs_partial),
    "layer_effect_on_ephys": {
        prop: {"r": round(pointbiserialr(layer_vec, get_prop_vals(prop, common_regions))[0], 4),
               "p": round(pointbiserialr(layer_vec, get_prop_vals(prop, common_regions))[1], 6)}
        for prop in ["burst_mean", "rate_mean", "nw_frac"]
    },
    "top_genes_layer_association": [
        {"gene": g, **v} for g, v in sorted_layer[:15]
    ],
    "partial_fdr_sig_list": [
        {"gene": v_gene, "prop": v_prop, "rho_partial": round(all_corrs_partial[k]["rho"], 4),
         "p_partial": round(all_corrs_partial[k]["p"], 6),
         "p_corr": round(pcorr_partial[i], 6),
         "rho_raw": round(all_corrs_raw[k]["rho"], 4)}
        for i, (k, v_partial) in enumerate(all_corrs_partial.items())
        if reject_partial[i]
        for v_gene, v_prop in [k.split("→")]
    ],
    "notes": f"{n_survive}/11 H24 FDR pairs survive p<0.05 after controlling for layer; {n_direction}/11 direction-consistent across layers; {n_partial_fdr}/{len(all_corrs_partial)} FDR-sig in full partial screen"
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
genes_h24 = [r["gene"] for r in partial_results]
x_raw = [abs(r["rho_raw"]) for r in partial_results]
x_partial = [abs(r["rho_partial"]) for r in partial_results]
y_pos = np.arange(len(genes_h24))
ax.barh(y_pos - 0.15, x_raw, 0.3, label="Raw", color="#4C72B0", alpha=0.8)
ax.barh(y_pos + 0.15, x_partial, 0.3, label="Partial (layer ctrl)", color="#DD8452", alpha=0.8)
ax.set_yticks(y_pos)
ax.set_yticklabels([f"{r['gene']}→{r['prop']}" for r in partial_results], fontsize=7)
ax.set_xlabel("|Spearman ρ|")
ax.set_title("H24 FDR-sig pairs: Raw vs Partial ρ")
ax.legend(fontsize=8)
ax.axvline(0, color="black", linestyle="--", alpha=0.3)

ax = axes[0, 1]
rho_drops = [r["rho_raw"] - r["rho_partial"] for r in partial_results]
ax.barh(y_pos, rho_drops, color=["#C44E52" if d > 0 else "#55A868" for d in rho_drops], alpha=0.8)
ax.set_yticks(y_pos)
ax.set_yticklabels([f"{r['gene']}→{r['prop']}" for r in partial_results], fontsize=7)
ax.set_xlabel("Δρ (raw - partial)")
ax.set_title("Attenuation after controlling for layer")
ax.axvline(0, color="black", linestyle="--", alpha=0.3)

ax = axes[1, 0]
l5_rhos = [r["rho_l5"] for r in partial_results if r["rho_l5"] is not None]
l6_rhos = [r["rho_l6"] for r in partial_results if r["rho_l6"] is not None]
labels_both = [f"{r['gene']}→{r['prop']}" for r in partial_results
               if r["rho_l5"] is not None and r["rho_l6"] is not None]
l5_both = [r["rho_l5"] for r in partial_results if r["rho_l5"] is not None and r["rho_l6"] is not None]
l6_both = [r["rho_l6"] for r in partial_results if r["rho_l5"] is not None and r["rho_l6"] is not None]
y2 = np.arange(len(labels_both))
ax.barh(y2 - 0.15, l5_both, 0.3, label="L5 (n=5)", color="#4C72B0", alpha=0.8)
ax.barh(y2 + 0.15, l6_both, 0.3, label="L6 (n=5)", color="#DD8452", alpha=0.8)
ax.set_yticks(y2)
ax.set_yticklabels(labels_both, fontsize=7)
ax.set_xlabel("Spearman ρ")
ax.set_title("Within-layer correlations (L5 vs L6)")
ax.legend(fontsize=8)
ax.axvline(0, color="black", linestyle="--", alpha=0.3)

ax = axes[1, 1]
layer_genes = sorted_layer[:15]
ax.barh(np.arange(len(layer_genes)), [g[1]["r"] for g in layer_genes],
        color=["#C44E52" if g[1]["r"] > 0 else "#55A868" for g in layer_genes], alpha=0.8)
ax.set_yticks(np.arange(len(layer_genes)))
ax.set_yticklabels([g[0] for g in layer_genes], fontsize=8)
ax.set_xlabel("Point-biserial r (L5=0, L6=1)")
ax.set_title("Gene expression vs Layer")
ax.axvline(0, color="black", linestyle="--", alpha=0.3)

plt.suptitle("H25: Layer-controlled gene→burst correlations", fontsize=14)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "figures" / "layer_controlled_gene_burst.png", dpi=150, bbox_inches="tight")
plt.savefig(OUTPUT_DIR / "figures" / "layer_controlled_gene_burst.svg", bbox_inches="tight")

print(f"\nResults saved to {out_path}")
print("Done.")

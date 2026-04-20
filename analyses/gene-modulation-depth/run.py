"""
Hypothesis H26: MERFISH gene expression (with layer control) predicts neural
modulation depth (M_mean) across PFC subregions — extending the gene→ephys
link to stimulus responsiveness.

M_mean is the modulation index: how much a neuron's activity changes in
response to stimulation. Negative = suppressed, positive = enhanced.

If CONFIRMED: modulation depth is gene-tractable (like burst index).
If REFUTED: modulation is not predictable from gene expression at region level.

Branch: gene-modulation-depth
Datasets: merfish, neural_activity
"""

import json
import random
from pathlib import Path

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from scipy.stats import spearmanr, pearsonr, pointbiserialr

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
print("H26: Gene expression → Neural modulation depth (M_mean)")
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

print("\n[2] Computing mean expression per subregion (all + Glut/GABA)...")
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

print("\n[3] Loading neural activity and computing M_mean per region...")
neural = pl.read_csv(DATA_PATH / "pfcmap" / "units.csv")
neural = neural.filter(pl.col("region").is_in(ALL_REGIONS))
neural = neural.filter(pl.col("utype").is_in(["ww", "nw"]))
neural = neural.filter(pl.col("rate_mean").is_not_null())
print(f"  Total neurons: {len(neural)}")
print(f"  M_mean non-null: {neural.filter(pl.col('M_mean').is_not_null()).shape[0]}")

neural_props = {}
for region in ALL_REGIONS:
    sub = neural.filter(pl.col("region") == region)
    sub_m = sub.filter(pl.col("M_mean").is_not_null())
    if len(sub_m) < 5:
        continue
    ww_m = sub_m.filter(pl.col("utype") == "ww")
    nw_m = sub_m.filter(pl.col("utype") == "nw")
    neural_props[region] = {
        "M_mean": float(sub_m["M_mean"].mean()),
        "M_mean_ww": float(ww_m["M_mean"].mean()) if len(ww_m) > 0 else None,
        "M_mean_nw": float(nw_m["M_mean"].mean()) if len(nw_m) > 0 else None,
        "burst_mean": float(sub.filter(pl.col("B_mean").is_not_null())["B_mean"].mean()),
        "rate_mean": float(sub["rate_mean"].mean()),
        "nw_frac": len(sub.filter(pl.col("utype") == "nw")) / len(sub),
        "n_M": len(sub_m),
        "n_total": len(sub),
    }

common_regions = sorted(set(expr_per_region.keys()) & set(neural_props.keys()))
print(f"  Common regions: {len(common_regions)}")

layer_vec = np.array([0.0 if r in L5_REGIONS else 1.0 for r in common_regions])

print("\n  M_mean by region:")
for r in common_regions:
    v = neural_props[r]
    print(f"    {r}: M={v['M_mean']:+.4f}, M_ww={v['M_mean_ww']:+.4f}, M_nw={v['M_mean_nw']:+.4f}, n_M={v['n_M']}")

r_layer, p_layer = pointbiserialr(layer_vec, np.array([neural_props[r]["M_mean"] for r in common_regions]))
print(f"\n  Layer → M_mean: r={r_layer:+.3f}, p={p_layer:.4f}, R²={r_layer**2:.3f}")

print("\n[4] Gene → M_mean correlations (raw + partial)...")
eprop_list = ["M_mean", "M_mean_ww", "M_mean_nw"]
eprop_labels = {"M_mean": "Modulation (all)", "M_mean_ww": "Modulation (WW)", "M_mean_nw": "Modulation (NW)"}

all_corrs = {}

for gene in targets_present:
    for prop in eprop_list:
        gene_vals = np.array([expr_per_region[r][gene] for r in common_regions])
        prop_vals = np.array([neural_props[r][prop] for r in common_regions if neural_props[r][prop] is not None])
        if len(prop_vals) < len(common_regions):
            valid_regions = [r for r in common_regions if neural_props[r][prop] is not None]
            gene_vals = np.array([expr_per_region[r][gene] for r in valid_regions])
            lv = np.array([0.0 if r in L5_REGIONS else 1.0 for r in valid_regions])
        else:
            lv = layer_vec
            valid_regions = common_regions

        valid = ~(np.isnan(gene_vals) | np.isnan(prop_vals))
        gv, pv, llv = gene_vals[valid], prop_vals[valid], lv[valid]

        if len(gv) < 4 or np.std(gv) == 0 or np.std(pv) == 0:
            continue

        rho_raw, p_raw = spearmanr(gv, pv)
        rho_partial, p_partial = partial_spearman(gv, pv, llv)

        key = f"{gene}→{prop}"
        all_corrs[key] = {
            "gene": gene, "prop": prop,
            "rho_raw": round(float(rho_raw), 4),
            "p_raw": round(float(p_raw), 6),
            "rho_partial": round(float(rho_partial), 4),
            "p_partial": round(float(p_partial), 6),
        }

    for prop in ["M_mean"]:
        for nt in ["Glut", "GABA"]:
            gene_vals = np.array([expr_per_region_nt[nt][r][gene] for r in common_regions])
            prop_vals = np.array([neural_props[r][prop] for r in common_regions])
            valid = ~(np.isnan(gene_vals) | np.isnan(prop_vals))
            gv, pv, lv2 = gene_vals[valid], prop_vals[valid], layer_vec[valid]
            if len(gv) < 4 or np.std(gv) == 0 or np.std(pv) == 0:
                continue
            rho_raw, p_raw = spearmanr(gv, pv)
            rho_partial, p_partial = partial_spearman(gv, pv, lv2)
            key = f"{gene}→{prop}_{nt}"
            all_corrs[key] = {
                "gene": gene, "prop": f"{prop}_{nt}",
                "rho_raw": round(float(rho_raw), 4),
                "p_raw": round(float(p_raw), 6),
                "rho_partial": round(float(rho_partial), 4),
                "p_partial": round(float(p_partial), 6),
            }

from statsmodels.stats.multitest import multipletests

raw_pvals = [v["p_raw"] for v in all_corrs.values()]
partial_pvals = [v["p_partial"] for v in all_corrs.values()]

reject_raw, pcorr_raw, _, _ = multipletests(raw_pvals, alpha=0.05, method="fdr_bh")
reject_partial, pcorr_partial, _, _ = multipletests(partial_pvals, alpha=0.05, method="fdr_bh")

n_raw_fdr = int(np.sum(reject_raw))
n_partial_fdr = int(np.sum(reject_partial))

print(f"\n  Total tests: {len(all_corrs)}")
print(f"  Raw FDR-sig: {n_raw_fdr}")
print(f"  Partial FDR-sig: {n_partial_fdr}")

print(f"\n  Top 20 raw gene→M_mean:")
sorted_raw = sorted(all_corrs.items(), key=lambda x: x[1]["p_raw"])
for i, (key, val) in enumerate(sorted_raw[:20]):
    idx = list(all_corrs.keys()).index(key)
    sig = "**" if reject_raw[idx] else "*" if val["p_raw"] < 0.05 else ""
    print(f"    {key:40s}: ρ={val['rho_raw']:+.3f}, p={val['p_raw']:.4f}{sig}")

print(f"\n  Top 20 partial gene→M_mean:")
sorted_partial = sorted(all_corrs.items(), key=lambda x: x[1]["p_partial"])
for i, (key, val) in enumerate(sorted_partial[:20]):
    idx = list(all_corrs.keys()).index(key)
    sig = "**" if reject_partial[idx] else "*" if val["p_partial"] < 0.05 else ""
    delta = val["rho_partial"] - val["rho_raw"]
    print(f"    {key:40s}: ρ_raw={val['rho_raw']:+.3f} → ρ_partial={val['rho_partial']:+.3f} (Δ={delta:+.3f}), p={val['p_partial']:.4f}{sig}")

print("\n[5] Comparing M_mean predictability vs burst/rate/NW...")
props_compare = {"burst_mean": [], "rate_mean": [], "nw_frac": [], "M_mean": []}

for gene in targets_present:
    for prop in ["burst_mean", "rate_mean", "nw_frac", "M_mean"]:
        gene_vals = np.array([expr_per_region[r][gene] for r in common_regions])
        if prop == "M_mean":
            prop_vals = np.array([neural_props[r][prop] for r in common_regions])
        else:
            sub_df = neural.filter(pl.col("region").is_in(common_regions))
            prop_vals = np.array([neural_props[r][prop] for r in common_regions])

        valid = ~(np.isnan(gene_vals) | np.isnan(prop_vals))
        gv, pv = gene_vals[valid], prop_vals[valid]
        if len(gv) < 4 or np.std(gv) == 0 or np.std(pv) == 0:
            continue
        rho, p = spearmanr(gv, pv)
        props_compare[prop].append(abs(rho))

print(f"\n  Mean |ρ| across {len(targets_present)} genes:")
for prop, rhos in props_compare.items():
    if rhos:
        print(f"    {prop:15s}: mean |ρ|={np.mean(rhos):.3f}, max |ρ|={np.max(rhos):.3f}, n>|0.7|={sum(1 for r in rhos if r>0.7)}")

print("\n[6] WW vs NW modulation depth difference...")
for region in common_regions:
    v = neural_props[region]
    if v["M_mean_ww"] is not None and v["M_mean_nw"] is not None:
        diff = v["M_mean_nw"] - v["M_mean_ww"]
        print(f"  {region}: NW M={v['M_mean_nw']:+.4f}, WW M={v['M_mean_ww']:+.4f}, diff={diff:+.4f}")

nw_mod = np.array([neural_props[r]["M_mean_nw"] for r in common_regions if neural_props[r]["M_mean_nw"] is not None])
ww_mod = np.array([neural_props[r]["M_mean_ww"] for r in common_regions if neural_props[r]["M_mean_ww"] is not None])
from scipy.stats import wilcoxon
if len(nw_mod) >= 5:
    try:
        w_stat, w_p = wilcoxon(nw_mod, ww_mod)
        print(f"\n  Wilcoxon NW vs WW M_mean: stat={w_stat:.3f}, p={w_p:.4f}")
        print(f"  NW mean M={np.mean(nw_mod):+.4f}, WW mean M={np.mean(ww_mod):+.4f}")
    except Exception as e:
        print(f"  Wilcoxon test failed: {e}")

status = "CONFIRMED" if n_partial_fdr > 0 or n_raw_fdr > 0 else "REFUTED"

results = {
    "hypothesis_id": "H26",
    "status": status,
    "n_tests": len(all_corrs),
    "n_raw_fdr": n_raw_fdr,
    "n_partial_fdr": n_partial_fdr,
    "layer_effect_on_M_mean": {"r": round(float(r_layer), 4), "p": round(float(p_layer), 6), "R2": round(float(r_layer**2), 4)},
    "top_raw": [
        {"key": k, **v} for k, v in sorted_raw[:15]
    ],
    "top_partial": [
        {"key": k, **v} for k, v in sorted_partial[:15]
    ],
    "fdr_partial_sig": [
        {"key": k, "gene": v["gene"], "prop": v["prop"],
         "rho_raw": v["rho_raw"], "rho_partial": v["rho_partial"],
         "p_corr": round(float(pcorr_partial[i]), 6)}
        for i, (k, v) in enumerate(all_corrs.items()) if reject_partial[i]
    ],
    "fdr_raw_sig": [
        {"key": k, "gene": v["gene"], "prop": v["prop"],
         "rho_raw": v["rho_raw"], "p_corr": round(float(pcorr_raw[i]), 6)}
        for i, (k, v) in enumerate(all_corrs.items()) if reject_raw[i]
    ],
    "predictability_comparison": {
        prop: {"mean_abs_rho": round(float(np.mean(rhos)), 4), "max_abs_rho": round(float(np.max(rhos)), 4)}
        for prop, rhos in props_compare.items() if rhos
    },
    "notes": f"{n_partial_fdr}/{len(all_corrs)} partial FDR-sig, {n_raw_fdr}/{len(all_corrs)} raw FDR-sig; {status}"
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
top_raw_m = [(k, v) for k, v in sorted_raw if "M_mean" in v["prop"] and not v["prop"].endswith("_Glut") and not v["prop"].endswith("_GABA")][:20]
if top_raw_m:
    names = [v["gene"] for k, v in top_raw_m]
    rhos = [v["rho_raw"] for k, v in top_raw_m]
    pvals = [v["p_raw"] for k, v in top_raw_m]
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
    ax.barh(np.arange(len(names)), rhos, color=colors, edgecolor="black", alpha=0.8)
    ax.set_yticks(np.arange(len(names)))
    ax.set_yticklabels(names, fontsize=8)
    ax.axvline(0, color="black", linestyle="--", alpha=0.5)
    ax.set_xlabel("Spearman ρ")
    ax.set_title(f"Raw: Genes → M_mean (n={len(common_regions)})")

ax = axes[0, 1]
top_partial_m = [(k, v) for k, v in sorted_partial if "M_mean" in v["prop"] and not v["prop"].endswith("_Glut") and not v["prop"].endswith("_GABA")][:20]
if top_partial_m:
    names = [v["gene"] for k, v in top_partial_m]
    rhos = [v["rho_partial"] for k, v in top_partial_m]
    rhos_raw = [v["rho_raw"] for k, v in top_partial_m]
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
    ax.barh(np.arange(len(names)), rhos, color=colors, edgecolor="black", alpha=0.8)
    ax.set_yticks(np.arange(len(names)))
    ax.set_yticklabels(names, fontsize=8)
    ax.axvline(0, color="black", linestyle="--", alpha=0.5)
    ax.set_xlabel("Partial Spearman ρ (layer ctrl)")
    ax.set_title(f"Partial: Genes → M_mean (n={len(common_regions)})")

ax = axes[1, 0]
props_list = list(props_compare.keys())
means = [np.mean(props_compare[p]) if props_compare[p] else 0 for p in props_list]
maxs = [np.max(props_compare[p]) if props_compare[p] else 0 for p in props_list]
x = np.arange(len(props_list))
ax.bar(x - 0.15, means, 0.3, label="Mean |ρ|", color="#4C72B0", alpha=0.8)
ax.bar(x + 0.15, maxs, 0.3, label="Max |ρ|", color="#DD8452", alpha=0.8)
ax.set_xticks(x)
ax.set_xticklabels(props_list, fontsize=9)
ax.set_ylabel("|Spearman ρ|")
ax.set_title("Gene predictability by ephys property")
ax.legend()

ax = axes[1, 1]
M_vals = [neural_props[r]["M_mean"] for r in common_regions]
burst_vals = [neural_props[r]["burst_mean"] for r in common_regions]
ax.scatter(burst_vals, M_vals, c=["#4C72B0" if r in L5_REGIONS else "#DD8452" for r in common_regions],
           s=80, edgecolors="black", alpha=0.8)
for i, r in enumerate(common_regions):
    ax.annotate(r, (burst_vals[i], M_vals[i]), fontsize=7, ha="left", va="bottom")
rho_bm, p_bm = spearmanr(burst_vals, M_vals)
ax.set_xlabel("Burst index")
ax.set_ylabel("Modulation depth (M_mean)")
ax.set_title(f"Burst vs Modulation: ρ={rho_bm:+.3f}, p={p_bm:.3f}")
ax.axhline(0, color="gray", linestyle="--", alpha=0.3)
ax.axvline(0, color="gray", linestyle="--", alpha=0.3)

from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor="#4C72B0", edgecolor="black", label="L5"),
    Patch(facecolor="#DD8452", edgecolor="black", label="L6"),
]
axes[1, 1].legend(handles=legend_elements, loc="best", fontsize=8)

plt.suptitle("H26: Gene expression → Neural modulation depth (M_mean)", fontsize=14)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "figures" / "gene_modulation_depth.png", dpi=150, bbox_inches="tight")
plt.savefig(OUTPUT_DIR / "figures" / "gene_modulation_depth.svg", bbox_inches="tight")

print(f"\nResults saved to {out_path}")
print("Done.")

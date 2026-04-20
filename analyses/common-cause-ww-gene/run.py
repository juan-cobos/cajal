"""
Hypothesis H30: Gene expression (PC2 burst module) is a common cause of
both connectivity profiles and electrophysiology, not a mediator between them.
Additionally: gene expression predicts WW-specific electrophysiology (firing
rate, burst index, modulation depth) across PFC subregions.

Part A: Common-cause model — GenePC2 explains both ConnPC1 and burst, and
  Conn→Burst correlation vanishes when GenePC2 is controlled (replicating H29).
  But GenePC2→Conn and GenePC2→Burst should BOTH remain significant when the
  other is controlled, if GenePC2 is the upstream common cause.

Part B: WW-specific gene→ephys — WW neurons are the dominant population (~85%).
  Previous analyses aggregated all neurons. WW-specific properties (rate, burst,
  modulation) may show different gene predictability patterns.

Branch: common-cause-ww-gene
Datasets: connectivity, merfish, neural_activity
"""

import glob
import json
import random
from pathlib import Path

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from scipy.stats import spearmanr, pearsonr, rankdata, pointbiserialr
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


def partial_corr(x, y, *covariates):
    rx = rankdata(x)
    ry = rankdata(y)
    rcovs = [rankdata(c) for c in covariates]
    X = np.column_stack([np.ones(len(rx))] + rcovs)
    beta_x = np.linalg.lstsq(X, rx, rcond=None)[0]
    res_x = rx - X @ beta_x
    beta_y = np.linalg.lstsq(X, ry, rcond=None)[0]
    res_y = ry - X @ beta_y
    r, p = pearsonr(res_x, res_y)
    return float(r), float(p)


def spearman_partial(x, y, *covariates):
    rho, p = spearmanr(x, y)
    rho_p, p_p = partial_corr(x, y, *covariates)
    return float(rho), float(p), float(rho_p), float(p_p)


print("=" * 60)
print("H30: Common-cause model + WW-specific gene→ephys")
print("=" * 60)

print("\n[1] Loading MERFISH expression...")
adata = ad.read_h5ad(DATA_PATH / "merfish" / "C57BL6J-638850.h5ad")
meta = pl.read_csv(DATA_PATH / "merfish" / "cell_metadata_with_parcellation_annotation.csv")
pfc_cells = meta.filter(pl.col("parcellation_substructure").is_in(ALL_REGIONS))
cell_labels = set(pfc_cells["cell_label"].to_list())
adata = adata[adata.obs_names.isin(cell_labels)].copy()
obs_df = pfc_cells.to_pandas().set_index("cell_label")
common_cells = adata.obs_names.intersection(obs_df.index)
adata = adata[common_cells].copy()
adata.obs["parcellation_substructure"] = obs_df.loc[common_cells, "parcellation_substructure"].values
adata.obs["neurotransmitter"] = obs_df.loc[common_cells, "neurotransmitter"].values

gene_symbols = adata.var["gene_symbol"].to_dict()
available_genes = {v: k for k, v in gene_symbols.items()}
targets_present = [g for g in ALL_TARGETS if g in available_genes]

gene_expr = {}
for gene in targets_present:
    gene_idx = available_genes[gene]
    gene_expr[gene] = np.array(adata[:, gene_idx].X.todense()).flatten()

subregions = adata.obs["parcellation_substructure"].values
nt_values = adata.obs["neurotransmitter"].values

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
        nt_mask = mask & (nt_values == nt)
        for gene in targets_present:
            expr_per_region_nt[nt][region][gene] = float(gene_expr[gene][nt_mask].mean())

gene_matrix = np.array([[expr_per_region[r][g] for g in targets_present] for r in ALL_REGIONS])

print("\n[2] Gene expression PCA...")
gene_z = (gene_matrix - gene_matrix.mean(axis=0)) / (gene_matrix.std(axis=0) + 1e-10)
gene_z = np.nan_to_num(gene_z, nan=0.0)
pca_gene = PCA(n_components=5, random_state=SEED)
gene_pc = pca_gene.fit_transform(gene_z)
gene_pc1 = gene_pc[:, 0]
gene_pc2 = gene_pc[:, 1]

print(f"  GenePC1 (layer axis): {pca_gene.explained_variance_ratio_[0]:.3f}")
print(f"  GenePC2 (burst module): {pca_gene.explained_variance_ratio_[1]:.3f}")

print("\n[3] Building connectivity PCA...")
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

conn_z = (conn_strong - conn_strong.mean(axis=0)) / (conn_strong.std(axis=0) + 1e-10)
conn_z = np.nan_to_num(conn_z, nan=0.0)
pca_conn = PCA(n_components=5, random_state=SEED)
conn_pc = pca_conn.fit_transform(conn_z)
conn_pc1 = conn_pc[:, 0]

layer_vec = np.array([0.0 if r in L5_REGIONS else 1.0 for r in ALL_REGIONS])

print("\n[4] Loading neural activity — WW-specific properties...")
neural = pl.read_csv(DATA_PATH / "pfcmap" / "units.csv")
neural = neural.filter(pl.col("region").is_in(ALL_REGIONS))
neural = neural.filter(pl.col("utype").is_in(["ww", "nw"]))
neural = neural.filter(pl.col("rate_mean").is_not_null())

ww_ephys = {}
for region in ALL_REGIONS:
    sub = neural.filter(pl.col("region") == region)
    ww = sub.filter(pl.col("utype") == "ww")
    nw = sub.filter(pl.col("utype") == "nw")
    ww_b = ww.filter(pl.col("B_mean").is_not_null())
    ww_m = ww.filter(pl.col("M_mean").is_not_null())
    nw_b = nw.filter(pl.col("B_mean").is_not_null())

    ww_ephys[region] = {
        "burst_all": float(sub.filter(pl.col("B_mean").is_not_null())["B_mean"].mean()),
        "rate_all": float(sub["rate_mean"].mean()),
        "nw_frac": len(nw) / len(sub),
        "ww_rate": float(ww["rate_mean"].mean()),
        "ww_burst": float(ww_b["B_mean"].mean()) if len(ww_b) > 0 else None,
        "ww_M": float(ww_m["M_mean"].mean()) if len(ww_m) > 0 else None,
        "nw_rate": float(nw["rate_mean"].mean()) if len(nw) > 0 else None,
        "nw_burst": float(nw_b["B_mean"].mean()) if len(nw_b) > 0 else None,
        "n_ww": len(ww),
        "n_nw": len(nw),
    }

print("  WW ephys per region:")
for r in ALL_REGIONS:
    v = ww_ephys[r]
    print(f"    {r}: ww_rate={v['ww_rate']:.3f}, ww_burst={v['ww_burst']:+.4f}, ww_M={v['ww_M']:+.4f}")

# ============================================================
# PART A: Common-cause model
# ============================================================
print("\n" + "=" * 60)
print("PART A: Common-cause model")
print("=" * 60)

burst_all = np.array([ww_ephys[r]["burst_all"] for r in ALL_REGIONS])
rate_all = np.array([ww_ephys[r]["rate_all"] for r in ALL_REGIONS])

print("\n[A1] Testing three competing models:")

print("\n  Model 1 (mediation): Conn → Gene → Burst")
print("  Model 2 (common cause): Gene → Conn AND Gene → Burst")
print("  Model 3 (reverse): Burst → Gene → Conn")

print("\n  Zero-order correlations:")
pairs = [
    ("GenePC2", "Burst", gene_pc2, burst_all),
    ("GenePC2", "ConnPC1", gene_pc2, conn_pc1),
    ("ConnPC1", "Burst", conn_pc1, burst_all),
    ("GenePC2", "Rate", gene_pc2, rate_all),
    ("ConnPC1", "Rate", conn_pc1, rate_all),
    ("GenePC2", "Layer", gene_pc2, layer_vec),
    ("ConnPC1", "Layer", conn_pc1, layer_vec),
]
for name1, name2, v1, v2 in pairs:
    rho, p = spearmanr(v1, v2)
    print(f"    {name1} ↔ {name2}: ρ={rho:+.4f}, p={p:.4f}")

print("\n  Partial correlations (controlling for layer):")
for name1, name2, v1, v2 in pairs:
    rho_p, p_p = partial_corr(v1, v2, layer_vec)
    print(f"    {name1} ↔ {name2} (ctrl layer): ρ={rho_p:+.4f}, p={p_p:.4f}")

print("\n[A2] Key tests for common cause vs mediation:")
print("  If GenePC2 is common cause, both GenePC2→Burst and GenePC2→Conn")
print("  should remain significant when controlling for the other.")

rho_gb_ctrl_c, p_gb_ctrl_c = partial_corr(gene_pc2, burst_all, conn_pc1, layer_vec)
rho_gc_ctrl_b, p_gc_ctrl_b = partial_corr(gene_pc2, conn_pc1, burst_all, layer_vec)
rho_cb_ctrl_g, p_cb_ctrl_g = partial_corr(conn_pc1, burst_all, gene_pc2, layer_vec)

print(f"\n  GenePC2 → Burst (ctrl ConnPC1+layer): ρ={rho_gb_ctrl_c:+.4f}, p={p_gb_ctrl_c:.4f}")
print(f"  GenePC2 → ConnPC1 (ctrl Burst+layer): ρ={rho_gc_ctrl_b:+.4f}, p={p_gc_ctrl_b:.4f}")
print(f"  ConnPC1 → Burst (ctrl GenePC2+layer): ρ={rho_cb_ctrl_g:+.4f}, p={p_cb_ctrl_g:.4f}")

gene_survives_burst = p_gb_ctrl_c < 0.05
gene_survives_conn = p_gc_ctrl_b < 0.05
conn_survives = p_cb_ctrl_g < 0.05

if gene_survives_burst and gene_survives_conn and not conn_survives:
    common_cause = True
    print("\n  → COMMON CAUSE model supported: GenePC2 explains both Conn and Burst independently")
elif gene_survives_burst and conn_survives:
    common_cause = False
    print("\n  → PARTIAL mediation/common cause: both GenePC2 and ConnPC1 have independent effects on Burst")
elif not gene_survives_burst and conn_survives:
    common_cause = False
    print("\n  → DIRECT model: ConnPC1 directly affects Burst, GenePC2 not needed")
else:
    common_cause = False
    print("\n  → INCONCLUSIVE: neither path survives all controls")

# ============================================================
# PART B: WW-specific gene→ephys
# ============================================================
print("\n" + "=" * 60)
print("PART B: WW-specific gene→ephys correlations")
print("=" * 60)

ww_rate = np.array([ww_ephys[r]["ww_rate"] for r in ALL_REGIONS])
ww_burst = np.array([ww_ephys[r]["ww_burst"] for r in ALL_REGIONS])
ww_M = np.array([ww_ephys[r]["ww_M"] for r in ALL_REGIONS])

ww_props = {
    "ww_rate": ("WW firing rate", ww_rate),
    "ww_burst": ("WW burst index", ww_burst),
    "ww_M": ("WW modulation depth", ww_M),
}
all_ww_props = {
    "burst_all": ("Burst (all)", burst_all),
    "rate_all": ("Rate (all)", rate_all),
    "nw_frac": ("NW fraction", np.array([ww_ephys[r]["nw_frac"] for r in ALL_REGIONS])),
    "ww_rate": ("WW rate", ww_rate),
    "ww_burst": ("WW burst", ww_burst),
    "ww_M": ("WW M_mean", ww_M),
    "nw_rate": ("NW rate", np.array([ww_ephys[r]["nw_rate"] for r in ALL_REGIONS])),
    "nw_burst": ("NW burst", np.array([ww_ephys[r]["nw_burst"] for r in ALL_REGIONS])),
}

print("\n[B1] Layer effects on WW properties:")
for prop, (label, vals) in ww_props.items():
    r, p = pointbiserialr(layer_vec, vals)
    print(f"  Layer → {label}: r={r:+.4f}, p={p:.4f}, R²={r**2:.4f}")

print("\n[B2] Individual gene → WW property correlations (raw + partial)...")

ww_gene_corrs = {}
for gene in targets_present:
    for prop, (label, vals) in all_ww_props.items():
        gene_vals = np.array([expr_per_region[r][gene] for r in ALL_REGIONS])
        valid = ~(np.isnan(gene_vals) | np.isnan(vals))
        gv, pv, lv = gene_vals[valid], vals[valid], layer_vec[valid]
        if len(gv) < 4 or np.std(gv) == 0 or np.std(pv) == 0:
            continue
        rho_raw, p_raw = spearmanr(gv, pv)
        rho_partial, p_partial = partial_corr(gv, pv, lv)
        ww_gene_corrs[f"{gene}→{prop}"] = {
            "gene": gene, "prop": prop,
            "rho_raw": round(float(rho_raw), 4),
            "p_raw": round(float(p_raw), 6),
            "rho_partial": round(float(rho_partial), 4),
            "p_partial": round(float(p_partial), 6),
        }

    for prop_base in ["ww_burst", "ww_rate"]:
        for nt in ["Glut", "GABA"]:
            base_vals = all_ww_props[prop_base][1]
            gene_vals = np.array([expr_per_region_nt[nt][r][gene] for r in ALL_REGIONS])
            valid = ~(np.isnan(gene_vals) | np.isnan(base_vals))
            gv, pv, lv = gene_vals[valid], base_vals[valid], layer_vec[valid]
            if len(gv) < 4 or np.std(gv) == 0 or np.std(pv) == 0:
                continue
            rho_raw, p_raw = spearmanr(gv, pv)
            rho_partial, p_partial = partial_corr(gv, pv, lv)
            ww_gene_corrs[f"{gene}→{prop_base}_{nt}"] = {
                "gene": gene, "prop": f"{prop_base}_{nt}",
                "rho_raw": round(float(rho_raw), 4),
                "p_raw": round(float(p_raw), 6),
                "rho_partial": round(float(rho_partial), 4),
                "p_partial": round(float(p_partial), 6),
            }

from statsmodels.stats.multitest import multipletests

partial_pvals = [v["p_partial"] for v in ww_gene_corrs.values()]
raw_pvals = [v["p_raw"] for v in ww_gene_corrs.values()]

reject_raw, pcorr_raw, _, _ = multipletests(raw_pvals, alpha=0.05, method="fdr_bh")
reject_partial, pcorr_partial, _, _ = multipletests(partial_pvals, alpha=0.05, method="fdr_bh")

n_raw_fdr = int(np.sum(reject_raw))
n_partial_fdr = int(np.sum(reject_partial))

print(f"\n  Total tests: {len(ww_gene_corrs)}")
print(f"  Raw FDR-sig: {n_raw_fdr}")
print(f"  Partial FDR-sig: {n_partial_fdr}")

print(f"\n[B3] Top 20 partial gene→WW correlations:")
sorted_ww = sorted(ww_gene_corrs.items(), key=lambda x: x[1]["p_partial"])
for i, (key, val) in enumerate(sorted_ww[:20]):
    idx = list(ww_gene_corrs.keys()).index(key)
    sig = "**" if reject_partial[idx] else "*" if val["p_partial"] < 0.05 else ""
    delta = val["rho_partial"] - val["rho_raw"]
    print(f"    {key:40s}: ρ_raw={val['rho_raw']:+.3f} → ρ_partial={val['rho_partial']:+.3f} (Δ={delta:+.3f}), p={val['p_partial']:.4f}{sig}")

if n_partial_fdr > 0:
    print(f"\n[B4] All partial FDR-significant ({n_partial_fdr}):")
    for i, (key, val) in enumerate(ww_gene_corrs.items()):
        if reject_partial[i]:
            print(f"    {key:40s}: ρ_partial={val['rho_partial']:+.3f}, p_corr={pcorr_partial[i]:.4f}")

print(f"\n[B5] WW vs all-neuron predictability comparison:")
for prop_key in ["ww_rate", "rate_all", "ww_burst", "burst_all", "ww_M", "nw_rate", "nw_burst"]:
    prop_corrs = [abs(v["rho_partial"]) for k, v in ww_gene_corrs.items()
                  if v["prop"] == prop_key and not prop_key.endswith("_Glut") and not prop_key.endswith("_GABA")]
    if prop_corrs:
        label = all_ww_props.get(prop_key, (prop_key, None))[0]
        print(f"  {prop_key:15s}: mean |ρ|={np.mean(prop_corrs):.3f}, max |ρ|={np.max(prop_corrs):.3f}, n>|0.7|={sum(1 for r in prop_corrs if r > 0.7)}")

print(f"\n[B6] GenePC2 → WW properties:")
for prop, (label, vals) in all_ww_props.items():
    rho, p = spearmanr(gene_pc2, vals)
    rho_p, p_p = partial_corr(gene_pc2, vals, layer_vec)
    print(f"  GenePC2 → {label}: ρ={rho:+.4f} p={p:.4f} | partial: ρ={rho_p:+.4f} p={p_p:.4f}")

print(f"\n[B7] ConnPC1 → WW properties:")
for prop, (label, vals) in all_ww_props.items():
    rho, p = spearmanr(conn_pc1, vals)
    rho_p, p_p = partial_corr(conn_pc1, vals, layer_vec)
    print(f"  ConnPC1 → {label}: ρ={rho:+.4f} p={p:.4f} | partial: ρ={rho_p:+.4f} p={p_p:.4f}")

status = "CONFIRMED" if (n_partial_fdr > 0 or (gene_survives_burst and gene_survives_conn)) else "REFUTED"

results = {
    "hypothesis_id": "H30",
    "status": status,
    "part_a_common_cause": {
        "gene_pc2_to_burst_ctrl_conn_layer": {"rho": round(rho_gb_ctrl_c, 4), "p": round(p_gb_ctrl_c, 6)},
        "gene_pc2_to_conn_ctrl_burst_layer": {"rho": round(rho_gc_ctrl_b, 4), "p": round(p_gc_ctrl_b, 6)},
        "conn_to_burst_ctrl_gene_layer": {"rho": round(rho_cb_ctrl_g, 4), "p": round(p_cb_ctrl_g, 6)},
        "common_cause_supported": common_cause,
    },
    "part_b_ww_gene": {
        "n_total_tests": len(ww_gene_corrs),
        "n_raw_fdr": n_raw_fdr,
        "n_partial_fdr": n_partial_fdr,
        "fdr_partial_sig": [
            {"key": k, "rho_partial": v["rho_partial"], "p_corr": round(float(pcorr_partial[i]), 6)}
            for i, (k, v) in enumerate(ww_gene_corrs.items()) if reject_partial[i]
        ],
        "fdr_raw_sig": [
            {"key": k, "rho_raw": v["rho_raw"], "p_corr": round(float(pcorr_raw[i]), 6)}
            for i, (k, v) in enumerate(ww_gene_corrs.items()) if reject_raw[i]
        ],
        "ww_predictability": {
            prop: {"mean_abs_rho": round(float(np.mean([abs(v["rho_partial"]) for k, v in ww_gene_corrs.items() if v["prop"] == prop])), 4),
                   "max_abs_rho": round(float(np.max([abs(v["rho_partial"]) for k, v in ww_gene_corrs.items() if v["prop"] == prop])), 4)}
            for prop in ["ww_rate", "rate_all", "ww_burst", "burst_all", "ww_M", "nw_rate", "nw_burst"]
        },
    },
    "gene_pc2_to_ww": {
        prop: {"rho": round(float(spearmanr(gene_pc2, vals)[0]), 4),
               "rho_partial": round(float(partial_corr(gene_pc2, vals, layer_vec)[0]), 4)}
        for prop, (_, vals) in all_ww_props.items()
    },
    "notes": f"Common cause: {common_cause}; WW gene FDR: {n_partial_fdr}/{len(ww_gene_corrs)} partial; {status}"
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

fig, axes = plt.subplots(2, 3, figsize=(20, 12))

ax = axes[0, 0]
ax.scatter(gene_pc2, burst_all, c=["#4C72B0" if r in L5_REGIONS else "#DD8452" for r in ALL_REGIONS],
           s=100, alpha=0.8, edgecolors="black")
for i, r in enumerate(ALL_REGIONS):
    ax.annotate(r, (gene_pc2[i], burst_all[i]), fontsize=7)
ax.set_xlabel("GenePC2")
ax.set_ylabel("Burst index (all)")
rho, p = spearmanr(gene_pc2, burst_all)
ax.set_title(f"GenePC2 → Burst\nρ={rho:+.3f}, p={p:.3f}")

ax = axes[0, 1]
ax.scatter(gene_pc2, conn_pc1, c=["#4C72B0" if r in L5_REGIONS else "#DD8452" for r in ALL_REGIONS],
           s=100, alpha=0.8, edgecolors="black")
for i, r in enumerate(ALL_REGIONS):
    ax.annotate(r, (gene_pc2[i], conn_pc1[i]), fontsize=7)
ax.set_xlabel("GenePC2")
ax.set_ylabel("ConnPC1")
rho, p = spearmanr(gene_pc2, conn_pc1)
ax.set_title(f"GenePC2 → Conn\nρ={rho:+.3f}, p={p:.3f}")

ax = axes[0, 2]
ax.scatter(conn_pc1, burst_all, c=["#4C72B0" if r in L5_REGIONS else "#DD8452" for r in ALL_REGIONS],
           s=100, alpha=0.8, edgecolors="black")
for i, r in enumerate(ALL_REGIONS):
    ax.annotate(r, (conn_pc1[i], burst_all[i]), fontsize=7)
ax.set_xlabel("ConnPC1")
ax.set_ylabel("Burst index (all)")
rho, p = spearmanr(conn_pc1, burst_all)
rho_p, _ = partial_corr(conn_pc1, burst_all, gene_pc2, layer_vec)
ax.set_title(f"Conn → Burst\nρ={rho:+.3f}, ρ'|GenePC2={rho_p:+.3f}")

ax = axes[1, 0]
ax.scatter(gene_pc2, ww_burst, c=["#4C72B0" if r in L5_REGIONS else "#DD8452" for r in ALL_REGIONS],
           s=100, alpha=0.8, edgecolors="black")
for i, r in enumerate(ALL_REGIONS):
    ax.annotate(r, (gene_pc2[i], ww_burst[i]), fontsize=7)
ax.set_xlabel("GenePC2")
ax.set_ylabel("WW burst index")
rho, p = spearmanr(gene_pc2, ww_burst)
ax.set_title(f"GenePC2 → WW Burst\nρ={rho:+.3f}, p={p:.3f}")

ax = axes[1, 1]
ax.scatter(gene_pc2, ww_rate, c=["#4C72B0" if r in L5_REGIONS else "#DD8452" for r in ALL_REGIONS],
           s=100, alpha=0.8, edgecolors="black")
for i, r in enumerate(ALL_REGIONS):
    ax.annotate(r, (gene_pc2[i], ww_rate[i]), fontsize=7)
ax.set_xlabel("GenePC2")
ax.set_ylabel("WW firing rate")
rho, p = spearmanr(gene_pc2, ww_rate)
ax.set_title(f"GenePC2 → WW Rate\nρ={rho:+.3f}, p={p:.3f}")

ax = axes[1, 2]
ww_predict = [np.mean([abs(v["rho_partial"]) for k, v in ww_gene_corrs.items() if v["prop"] == prop])
              for prop in ["ww_rate", "rate_all", "ww_burst", "burst_all", "ww_M", "nw_rate", "nw_burst"]]
ww_labels = ["WW rate", "All rate", "WW burst", "All burst", "WW M", "NW rate", "NW burst"]
ax.barh(range(len(ww_labels)), ww_predict, color=["#4C72B0", "#8ecae6", "#DD8452", "#f4a261", "#55A868", "#C44E52", "#e76f51"], alpha=0.8)
ax.set_yticks(range(len(ww_labels)))
ax.set_yticklabels(ww_labels)
ax.set_xlabel("Mean |partial ρ| across 88 genes")
ax.set_title("Gene predictability by property")

from matplotlib.patches import Patch
for ax in axes.flat:
    if ax.get_legend() is None:
        pass
legend_elements = [Patch(facecolor="#4C72B0", label="L5"), Patch(facecolor="#DD8452", label="L6")]
axes[0, 0].legend(handles=legend_elements, fontsize=8, loc="lower right")

plt.suptitle("H30: Common-cause model + WW gene→ephys", fontsize=14)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "figures" / "common_cause_ww_gene.png", dpi=150, bbox_inches="tight")
plt.savefig(OUTPUT_DIR / "figures" / "common_cause_ww_gene.svg", bbox_inches="tight")

print(f"\nResults saved to {out_path}")
print("Done.")

"""
Hypothesis H31: Gene expression PCs predict electrophysiology PCs across
PFC subregions, tested with permutation-based FDR that properly accounts for
the dependency structure in both gene and ephys variables.

Problem with H24-H30: FDR-BH on 440-1056 tests assumes near-independence,
but 88 genes → 9 independent dimensions and 7 ephys properties → 4
independent dimensions, giving only ~36 effective independent tests.
Standard FDR overcorrects by treating correlated tests as independent,
making it too liberal. Additionally, with n=10, very high ρ values are
easy to obtain by chance.

Fix: 
1. Reduce to orthogonal PCs on both sides (gene PCs × ephys PCs)
2. Use permutation-based FDR: for each of N perms, compute ALL genePC→ephysPC
   correlations, take max |ρ| across all tests, build empirical null
   distribution of the max statistic. This properly accounts for dependency.
3. Apply to both all-neuron and WW-specific ephys properties.

If CONFIRMED: gene→ephys links survive proper correction.
If REFUTED: previous findings were inflated by improper multiple testing.

Branch: permutation-fdr-gene-pcs
Datasets: merfish, neural_activity
"""

import glob
import json
import random
from pathlib import Path

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from scipy.stats import spearmanr, pearsonr, rankdata
from sklearn.decomposition import PCA

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

DATA_PATH = Path(__file__).parent.parent.parent / "data"
OUTPUT_DIR = Path(__file__).parent

L5_REGIONS = ["PL5", "ILA5", "ORBl5", "ORBm5", "ORBvl5"]
L6_REGIONS = ["PL6a", "ILA6a", "ORBl6a", "ORBm6a", "ORBvl6a"]
ALL_REGIONS = L5_REGIONS + L6_REGIONS

N_PERM = 10000

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

CATS = {}
for group, genes in TARGET_GENES.items():
    for g in genes:
        CATS[g] = group


def partial_corr_rank(x, y, *covariates):
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


print("=" * 60)
print("H31: Permutation-based FDR for gene→ephys PCs")
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
layer_vec = np.array([0.0 if r in L5_REGIONS else 1.0 for r in ALL_REGIONS])

print("\n[2] Computing gene expression PCs...")
gene_z = (gene_matrix - gene_matrix.mean(axis=0)) / (gene_matrix.std(axis=0) + 1e-10)
gene_z = np.nan_to_num(gene_z, nan=0.0)
pca_gene = PCA(n_components=9, random_state=SEED)
gene_pc = pca_gene.fit_transform(gene_z)
print(f"  Gene PCs: {gene_pc.shape[1]}, cumulative variance: {pca_gene.explained_variance_ratio_.sum():.3f}")
for i in range(gene_pc.shape[1]):
    print(f"    GenePC{i+1}: {pca_gene.explained_variance_ratio_[i]:.3f}")

print("\n[3] Loading neural activity — all + WW + NW properties...")
neural = pl.read_csv(DATA_PATH / "pfcmap" / "units.csv")
neural = neural.filter(pl.col("region").is_in(ALL_REGIONS))
neural = neural.filter(pl.col("utype").is_in(["ww", "nw"]))
neural = neural.filter(pl.col("rate_mean").is_not_null())

ephys_props_raw = {}
for region in ALL_REGIONS:
    sub = neural.filter(pl.col("region") == region)
    sub_b = sub.filter(pl.col("B_mean").is_not_null())
    ww = sub.filter(pl.col("utype") == "ww")
    nw = sub.filter(pl.col("utype") == "nw")
    ww_b = ww.filter(pl.col("B_mean").is_not_null())
    nw_b = nw.filter(pl.col("B_mean").is_not_null())
    ww_m = ww.filter(pl.col("M_mean").is_not_null())

    ephys_props_raw[region] = {
        "burst_all": float(sub_b["B_mean"].mean()),
        "ww_burst": float(ww_b["B_mean"].mean()),
        "rate_all": float(sub["rate_mean"].mean()),
        "ww_rate": float(ww["rate_mean"].mean()),
        "nw_frac": len(nw) / len(sub),
        "ww_M": float(ww_m["M_mean"].mean()) if len(ww_m) > 0 else np.nan,
    }

prop_names = ["burst_all", "ww_burst", "rate_all", "ww_rate", "nw_frac", "ww_M"]
ephys_matrix = np.array([[ephys_props_raw[r][p] for p in prop_names] for r in ALL_REGIONS])

print("\n[4] Computing ephys PCs...")
ephys_z = (ephys_matrix - ephys_matrix.mean(axis=0)) / (ephys_matrix.std(axis=0) + 1e-10)
ephys_z = np.nan_to_num(ephys_z, nan=0.0)
pca_ephys = PCA(n_components=min(4, ephys_z.shape[1]), random_state=SEED)
ephys_pc = pca_ephys.fit_transform(ephys_z)
print(f"  Ephys PCs: {ephys_pc.shape[1]}, cumulative variance: {pca_ephys.explained_variance_ratio_.sum():.3f}")
for i in range(ephys_pc.shape[1]):
    loadings = pca_ephys.components_[i]
    print(f"    EphysPC{i+1}: {pca_ephys.explained_variance_ratio_[i]:.3f}")
    for j, pn in enumerate(prop_names):
        if abs(loadings[j]) > 0.3:
            print(f"      {pn}: {loadings[j]:+.3f}")

n_gene_pcs = gene_pc.shape[1]
n_ephys_pcs = ephys_pc.shape[1]
n_total_tests = n_gene_pcs * n_ephys_pcs
print(f"\n  Total effective tests: {n_gene_pcs} × {n_ephys_pcs} = {n_total_tests}")

print("\n[5] Observed genePC → ephysPC correlations (Spearman)...")
obs_rhos = np.zeros((n_gene_pcs, n_ephys_pcs))
obs_ps = np.zeros((n_gene_pcs, n_ephys_pcs))
for i in range(n_gene_pcs):
    for j in range(n_ephys_pcs):
        rho, p = spearmanr(gene_pc[:, i], ephys_pc[:, j])
        obs_rhos[i, j] = rho
        obs_ps[i, j] = p
        if abs(rho) > 0.5:
            print(f"    GenePC{i+1} → EphysPC{j+1}: ρ={rho:+.4f}, p={p:.4f}")

print("\n[6] Permutation-based FDR (max-statistic method)...")
print(f"  Running {N_PERM} permutations...")

max_rhos_perm = np.zeros(N_PERM)
perm_rhos_all = np.zeros((N_PERM, n_gene_pcs, n_ephys_pcs))

for perm in range(N_PERM):
    perm_idx = np.random.permutation(len(ALL_REGIONS))
    max_rho = 0.0
    for i in range(n_gene_pcs):
        for j in range(n_ephys_pcs):
            rho_p, _ = spearmanr(gene_pc[perm_idx, i], ephys_pc[:, j])
            perm_rhos_all[perm, i, j] = rho_p
            if abs(rho_p) > max_rho:
                max_rho = abs(rho_p)
    max_rhos_perm[perm] = max_rho

# Family-wise error rate (FWER) thresholds
fwer_05 = np.percentile(max_rhos_perm, 95)
fwer_01 = np.percentile(max_rhos_perm, 99)
fwer_001 = np.percentile(max_rhos_perm, 99.9)

print(f"\n  FWER-corrected thresholds:")
print(f"    α=0.05: |ρ| > {fwer_05:.4f}")
print(f"    α=0.01: |ρ| > {fwer_01:.4f}")
print(f"    α=0.001: |ρ| > {fwer_001:.4f}")

# Permutation-based FDR: for each test, compute FDR as proportion of perms
# where the max |rho| >= observed |rho|
fdr_sig = []
for i in range(n_gene_pcs):
    for j in range(n_ephys_pcs):
        obs_abs = abs(obs_rhos[i, j])
        fdr_p = (np.sum(max_rhos_perm >= obs_abs) + 1) / (N_PERM + 1)

        # Also compute pointwise permutation p
        perm_vals = perm_rhos_all[:, i, j]
        pointwise_p = (np.sum(np.abs(perm_vals) >= obs_abs) + 1) / (N_PERM + 1)

        fdr_sig.append({
            "gene_pc": i + 1,
            "ephys_pc": j + 1,
            "rho": round(float(obs_rhos[i, j]), 4),
            "p_uncorrected": round(float(obs_ps[i, j]), 6),
            "p_pointwise_perm": round(float(pointwise_p), 6),
            "p_fwer": round(float(fdr_p), 6),
            "fwer_sig_05": obs_abs >= fwer_05,
            "fwer_sig_01": obs_abs >= fwer_01,
        })

fdr_sig.sort(key=lambda x: x["p_fwer"])

print(f"\n  All genePC → ephysPC correlations with permutation correction:")
print(f"  {'Test':20s} {'ρ':>8s} {'p_uncorr':>10s} {'p_perm':>10s} {'p_FWER':>10s} {'FWER_05':>8s}")
for entry in fdr_sig:
    sig = "***" if entry["fwer_sig_01"] else "**" if entry["fwer_sig_05"] else ""
    print(f"  GenePC{entry['gene_pc']}→EphysPC{entry['ephys_pc']} {entry['rho']:+8.4f} {entry['p_uncorrected']:10.6f} {entry['p_pointwise_perm']:10.6f} {entry['p_fwer']:10.6f} {'YES' if entry['fwer_sig_05'] else 'no':>8s} {sig}")

n_fwer_05 = sum(1 for e in fdr_sig if e["fwer_sig_05"])
n_fwer_01 = sum(1 for e in fdr_sig if e["fwer_sig_01"])
print(f"\n  FWER α=0.05 significant: {n_fwer_05}/{n_total_tests}")
print(f"  FWER α=0.01 significant: {n_fwer_01}/{n_total_tests}")

print("\n[7] Now: partial correlation (controlling for layer) with permutation FDR...")
print(f"  Computing partial ρ for {n_gene_pcs}×{n_ephys_pcs} = {n_total_tests} tests...")

obs_rhos_partial = np.zeros((n_gene_pcs, n_ephys_pcs))
for i in range(n_gene_pcs):
    for j in range(n_ephys_pcs):
        rho_p, _ = partial_corr_rank(gene_pc[:, i], ephys_pc[:, j], layer_vec)
        obs_rhos_partial[i, j] = rho_p

# Permutation for partial: permute gene PCs, recompute partial correlation
print(f"  Running {N_PERM} permutations for partial correlations...")

max_rhos_partial_perm = np.zeros(N_PERM)
for perm in range(N_PERM):
    perm_idx = np.random.permutation(len(ALL_REGIONS))
    max_rho = 0.0
    for i in range(n_gene_pcs):
        for j in range(n_ephys_pcs):
            rho_p, _ = partial_corr_rank(gene_pc[perm_idx, i], ephys_pc[:, j], layer_vec)
            if abs(rho_p) > max_rho:
                max_rho = abs(rho_p)
    max_rhos_partial_perm[perm] = max_rho

fwer_partial_05 = np.percentile(max_rhos_partial_perm, 95)
fwer_partial_01 = np.percentile(max_rhos_partial_perm, 99)

print(f"\n  Partial FWER thresholds:")
print(f"    α=0.05: |ρ| > {fwer_partial_05:.4f}")
print(f"    α=0.01: |ρ| > {fwer_partial_01:.4f}")

partial_results = []
for i in range(n_gene_pcs):
    for j in range(n_ephys_pcs):
        obs_abs = abs(obs_rhos_partial[i, j])
        fwer_p = (np.sum(max_rhos_partial_perm >= obs_abs) + 1) / (N_PERM + 1)
        partial_results.append({
            "gene_pc": i + 1,
            "ephys_pc": j + 1,
            "rho_partial": round(float(obs_rhos_partial[i, j]), 4),
            "rho_raw": round(float(obs_rhos[i, j]), 4),
            "p_fwer": round(float(fwer_p), 6),
            "fwer_sig_05": obs_abs >= fwer_partial_05,
            "fwer_sig_01": obs_abs >= fwer_partial_01,
        })

partial_results.sort(key=lambda x: x["p_fwer"])

print(f"\n  All partial genePC → ephysPC (layer-controlled):")
print(f"  {'Test':20s} {'ρ_raw':>8s} {'ρ_partial':>10s} {'p_FWER':>10s} {'FWER_05':>8s}")
for entry in partial_results:
    sig = "***" if entry["fwer_sig_01"] else "**" if entry["fwer_sig_05"] else ""
    print(f"  GenePC{entry['gene_pc']}→EphysPC{entry['ephys_pc']} {entry['rho_raw']:+8.4f} {entry['rho_partial']:+10.4f} {entry['p_fwer']:10.6f} {'YES' if entry['fwer_sig_05'] else 'no':>8s} {sig}")

n_partial_fwer_05 = sum(1 for e in partial_results if e["fwer_sig_05"])
n_partial_fwer_01 = sum(1 for e in partial_results if e["fwer_sig_01"])
print(f"\n  Partial FWER α=0.05 significant: {n_partial_fwer_05}/{n_total_tests}")
print(f"  Partial FWER α=0.01 significant: {n_partial_fwer_01}/{n_total_tests}")

print("\n[8] Comparison: standard FDR-BH vs permutation FWER on same data...")

from statsmodels.stats.multitest import multipletests

# Standard BH on the 36 tests
raw_pvals_flat = obs_ps.flatten()
reject_bh, pcorr_bh, _, _ = multipletests(raw_pvals_flat, alpha=0.05, method="fdr_bh")
n_bh_sig = int(np.sum(reject_bh))

# Standard BH on the 440 tests (88 genes × 5 ephys props — like H24)
all_gene_ephys_pvals = []
for gene in targets_present:
    gene_vals = np.array([expr_per_region[r][gene] for r in ALL_REGIONS])
    for prop in prop_names:
        prop_vals = np.array([ephys_props_raw[r][prop] for r in ALL_REGIONS])
        valid = ~(np.isnan(gene_vals) | np.isnan(prop_vals))
        if valid.sum() < 4 or np.std(gene_vals[valid]) == 0 or np.std(prop_vals[valid]) == 0:
            all_gene_ephys_pvals.append(1.0)
            continue
        rho, p = spearmanr(gene_vals[valid], prop_vals[valid])
        all_gene_ephys_pvals.append(float(p))

reject_bh_440, _, _, _ = multipletests(all_gene_ephys_pvals, alpha=0.05, method="fdr_bh")
n_bh_440_sig = int(np.sum(reject_bh_440))

print(f"  Method comparison:")
print(f"    Standard BH on {len(all_gene_ephys_pvals)} individual tests: {n_bh_440_sig} sig")
print(f"    Standard BH on {n_total_tests} PC tests: {n_bh_sig} sig")
print(f"    Permutation FWER (α=0.05) on {n_total_tests} PC tests: {n_fwer_05} sig (raw)")
print(f"    Permutation FWER (α=0.05) on {n_total_tests} PC tests (partial): {n_partial_fwer_05} sig")

print("\n[9] Which gene PC loadings drive the FWER-significant results?")

for entry in partial_results:
    if entry["fwer_sig_05"]:
        pc_idx = entry["gene_pc"] - 1
        loadings = pca_gene.components_[pc_idx]
        sorted_idx = np.argsort(np.abs(loadings))[::-1]
        print(f"\n  GenePC{entry['gene_pc']} (→ EphysPC{entry['ephys_pc']}, ρ_partial={entry['rho_partial']:+.4f}):")
        print(f"    Top 10 gene loadings:")
        for idx in sorted_idx[:10]:
            g = targets_present[idx]
            print(f"      {g:12s} ({CATS.get(g, '?'):25s}): {loadings[idx]:+.4f}")

        ephys_pc_idx = entry["ephys_pc"] - 1
        ephys_loadings = pca_ephys.components_[ephys_pc_idx]
        print(f"    EphysPC{entry['ephys_pc']} loadings:")
        for j, pn in enumerate(prop_names):
            if abs(ephys_loadings[j]) > 0.2:
                print(f"      {pn:15s}: {ephys_loadings[j]:+.4f}")

status = "CONFIRMED" if n_partial_fwer_05 > 0 else "REFUTED"

results = {
    "hypothesis_id": "H31",
    "status": status,
    "n_regions": len(ALL_REGIONS),
    "n_gene_pcs": n_gene_pcs,
    "n_ephys_pcs": n_ephys_pcs,
    "n_effective_tests": n_total_tests,
    "gene_pc_variance": [round(float(v), 4) for v in pca_gene.explained_variance_ratio_],
    "ephys_pc_variance": [round(float(v), 4) for v in pca_ephys.explained_variance_ratio_],
    "ephys_pc_loadings": {
        f"EphysPC{i+1}": {prop_names[j]: round(float(pca_ephys.components_[i][j]), 4)
                           for j in range(len(prop_names))}
        for i in range(n_ephys_pcs)
    },
    "fwer_thresholds_raw": {"alpha_05": round(float(fwer_05), 4), "alpha_01": round(float(fwer_01), 4)},
    "fwer_thresholds_partial": {"alpha_05": round(float(fwer_partial_05), 4), "alpha_01": round(float(fwer_partial_01), 4)},
    "n_fwer_sig_raw_05": n_fwer_05,
    "n_fwer_sig_raw_01": n_fwer_01,
    "n_fwer_sig_partial_05": n_partial_fwer_05,
    "n_fwer_sig_partial_01": n_partial_fwer_01,
    "fwer_sig_partial": [e for e in partial_results if e["fwer_sig_05"]],
    "all_partial_results": partial_results,
    "comparison": {
        "bh_440_individual": n_bh_440_sig,
        "bh_36_pc": n_bh_sig,
        "fwer_36_pc_raw": n_fwer_05,
        "fwer_36_pc_partial": n_partial_fwer_05,
    },
    "notes": f"Permutation FWER α=0.05: {n_partial_fwer_05}/{n_total_tests} partial sig; {status}"
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

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

ax = axes[0]
im = ax.imshow(obs_rhos, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(n_ephys_pcs))
ax.set_xticklabels([f"EP{i+1}" for i in range(n_ephys_pcs)])
ax.set_yticks(range(n_gene_pcs))
ax.set_yticklabels([f"GP{i+1}" for i in range(n_gene_pcs)])
ax.set_xlabel("Ephys PCs")
ax.set_ylabel("Gene PCs")
ax.set_title(f"Raw Spearman ρ\n(FWER α=0.05: |ρ|>{fwer_05:.3f})")
plt.colorbar(im, ax=ax)
for i in range(n_gene_pcs):
    for j in range(n_ephys_pcs):
        color = "white" if abs(obs_rhos[i, j]) > 0.5 else "black"
        sig = "***" if abs(obs_rhos[i, j]) >= fwer_01 else "**" if abs(obs_rhos[i, j]) >= fwer_05 else ""
        ax.text(j, i, f"{obs_rhos[i,j]:+.2f}\n{sig}", ha="center", va="center", fontsize=7, color=color)

ax = axes[1]
im = ax.imshow(obs_rhos_partial, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(n_ephys_pcs))
ax.set_xticklabels([f"EP{i+1}" for i in range(n_ephys_pcs)])
ax.set_yticks(range(n_gene_pcs))
ax.set_yticklabels([f"GP{i+1}" for i in range(n_gene_pcs)])
ax.set_xlabel("Ephys PCs")
ax.set_ylabel("Gene PCs")
ax.set_title(f"Partial ρ (layer ctrl)\n(FWER α=0.05: |ρ|>{fwer_partial_05:.3f})")
plt.colorbar(im, ax=ax)
for i in range(n_gene_pcs):
    for j in range(n_ephys_pcs):
        color = "white" if abs(obs_rhos_partial[i, j]) > 0.5 else "black"
        sig = "***" if abs(obs_rhos_partial[i, j]) >= fwer_partial_01 else "**" if abs(obs_rhos_partial[i, j]) >= fwer_partial_05 else ""
        ax.text(j, i, f"{obs_rhos_partial[i,j]:+.2f}\n{sig}", ha="center", va="center", fontsize=7, color=color)

ax = axes[2]
max_perm_sorted = np.sort(max_rhos_partial_perm)
ax.hist(max_perm_sorted, bins=50, density=True, alpha=0.7, color="#4C72B0", label="Null max |ρ|")
ax.axvline(fwer_partial_05, color="#DD8452", linestyle="--", label=f"FWER α=0.05 ({fwer_partial_05:.3f})")
ax.axvline(fwer_partial_01, color="#C44E52", linestyle="--", label=f"FWER α=0.01 ({fwer_partial_01:.3f})")
for entry in partial_results:
    if entry["fwer_sig_05"]:
        ax.axvline(abs(entry["rho_partial"]), color="#55A868", linestyle="-", alpha=0.5)
ax.set_xlabel("Max |ρ| across 36 tests")
ax.set_ylabel("Density")
ax.set_title("Permutation null distribution\n(max |partial ρ|)")
ax.legend(fontsize=8)

plt.suptitle("H31: Permutation-based FWER for genePC→ephysPC", fontsize=14)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "figures" / "permutation_fdr_gene_pcs.png", dpi=150, bbox_inches="tight")
plt.savefig(OUTPUT_DIR / "figures" / "permutation_fdr_gene_pcs.svg", bbox_inches="tight")

print(f"\nResults saved to {out_path}")
print("Done.")

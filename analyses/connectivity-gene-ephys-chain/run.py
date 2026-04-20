"""
Hypothesis H29: Gene expression mediates the relationship between PFC
connectivity profiles and burst index — completing the 3-way chain
(connectivity → transcriptomics → electrophysiology).

H28 showed connectivity ↔ gene expression (r=+0.554, p=0.0003).
H25 showed gene → burst (layer-independent, ρ up to 0.90).
H27 showed PC2 (burst module) → burst (FDR-sig).

This hypothesis tests whether connectivity predicts burst index, and whether
gene expression mediates that relationship (partial mediation).

If CONFIRMED: connectivity→gene→burst chain is validated.
If REFUTED: connectivity and burst are not linked, or gene doesn't mediate.

Branch: connectivity-gene-ephys-chain
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
print("H29: Connectivity → Gene expression → Burst index chain")
print("=" * 60)

print("\n[1] Loading MERFISH expression...")
adata = ad.read_h5ad(DATA_PATH / "merfish" / "C57BL6J-638850.h5ad")
meta = pl.read_csv(DATA_PATH / "merfish" / "cell_metadata_with_parcellation_annotation.csv")
pfc_cells = meta.filter(pl.col("parcellation_substructure").is_in(ALL_REGIONS))
cell_labels = set(pfc_cells["cell_label"].to_list())
adata = adata[adata.obs_names.isin(cell_labels)].copy()
obs_df = pfc_cells.to_pandas().set_index("cell_label")
common = adata.obs_names.intersection(obs_df.index)
adata = adata[common].copy()
adata.obs["parcellation_substructure"] = obs_df.loc[common, "parcellation_substructure"].values

gene_symbols = adata.var["gene_symbol"].to_dict()
available_genes = {v: k for k, v in gene_symbols.items()}
targets_present = [g for g in ALL_TARGETS if g in available_genes]

gene_expr = {}
for gene in targets_present:
    gene_idx = available_genes[gene]
    gene_expr[gene] = np.array(adata[:, gene_idx].X.todense()).flatten()

subregions = adata.obs["parcellation_substructure"].values
expr_per_region = {}
for region in ALL_REGIONS:
    mask = subregions == region
    expr_per_region[region] = {}
    for gene in targets_present:
        expr_per_region[region][gene] = float(gene_expr[gene][mask].mean())

gene_matrix = np.array([[expr_per_region[r][g] for g in targets_present] for r in ALL_REGIONS])

print("\n[2] Building connectivity PCA scores...")
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
print(f"  Connectivity matrix: {conn_mat.shape}")

strong_mask = (conn_mat > 0.1).sum(axis=0) >= 3
conn_strong = conn_mat[:, strong_mask]
print(f"  Strong targets: {conn_strong.shape[1]}")

conn_z = (conn_strong - conn_strong.mean(axis=0)) / (conn_strong.std(axis=0) + 1e-10)
conn_z = np.nan_to_num(conn_z, nan=0.0)
pca_conn = PCA(n_components=min(5, conn_strong.shape[1]), random_state=SEED)
conn_pc = pca_conn.fit_transform(conn_z)
print(f"  Connectivity PC1 explains: {pca_conn.explained_variance_ratio_[0]:.3f}")
print(f"  Connectivity PC2 explains: {pca_conn.explained_variance_ratio_[1]:.3f}")

print("\n[3] Building gene expression PCA scores...")
gene_z = (gene_matrix - gene_matrix.mean(axis=0)) / (gene_matrix.std(axis=0) + 1e-10)
gene_z = np.nan_to_num(gene_z, nan=0.0)
pca_gene = PCA(n_components=5, random_state=SEED)
gene_pc = pca_gene.fit_transform(gene_z)
print(f"  Gene PC1 explains: {pca_gene.explained_variance_ratio_[0]:.3f}")
print(f"  Gene PC2 explains: {pca_gene.explained_variance_ratio_[1]:.3f}")

print("\n[4] Loading neural activity...")
neural = pl.read_csv(DATA_PATH / "pfcmap" / "units.csv")
neural = neural.filter(pl.col("region").is_in(ALL_REGIONS))
neural = neural.filter(pl.col("utype").is_in(["ww", "nw"]))
neural = neural.filter(pl.col("rate_mean").is_not_null())

neural_props = {}
for region in ALL_REGIONS:
    sub = neural.filter(pl.col("region") == region)
    ww = sub.filter(pl.col("utype") == "ww")
    nw = sub.filter(pl.col("utype") == "nw")
    neural_props[region] = {
        "burst_mean": float(sub.filter(pl.col("B_mean").is_not_null())["B_mean"].mean()),
        "rate_mean": float(sub["rate_mean"].mean()),
        "nw_frac": len(nw) / len(sub),
    }

layer_vec = np.array([0.0 if r in L5_REGIONS else 1.0 for r in ALL_REGIONS])
burst_vals = np.array([neural_props[r]["burst_mean"] for r in ALL_REGIONS])
rate_vals = np.array([neural_props[r]["rate_mean"] for r in ALL_REGIONS])
nw_vals = np.array([neural_props[r]["nw_frac"] for r in ALL_REGIONS])

print("\n[5] Testing direct connectivity → ephys links...")

print("\n  Connectivity PC scores → Burst index:")
for i in range(min(5, conn_pc.shape[1])):
    rho, p = spearmanr(conn_pc[:, i], burst_vals)
    rho_p, p_p = partial_spearman(conn_pc[:, i], burst_vals, layer_vec)
    print(f"    ConnPC{i+1} → burst: ρ={rho:+.4f} p={p:.4f} | partial: ρ={rho_p:+.4f} p={p_p:.4f}")

print("\n  Connectivity PC scores → Rate:")
for i in range(min(5, conn_pc.shape[1])):
    rho, p = spearmanr(conn_pc[:, i], rate_vals)
    rho_p, p_p = partial_spearman(conn_pc[:, i], rate_vals, layer_vec)
    print(f"    ConnPC{i+1} → rate: ρ={rho:+.4f} p={p:.4f} | partial: ρ={rho_p:+.4f} p={p_p:.4f}")

print("\n  Connectivity PC scores → NW fraction:")
for i in range(min(5, conn_pc.shape[1])):
    rho, p = spearmanr(conn_pc[:, i], nw_vals)
    rho_p, p_p = partial_spearman(conn_pc[:, i], nw_vals, layer_vec)
    print(f"    ConnPC{i+1} → NW: ρ={rho:+.4f} p={p:.4f} | partial: ρ={rho_p:+.4f} p={p_p:.4f}")

print("\n[6] Mediation analysis: Conn → GenePC2 → Burst...")

gene_pc2 = gene_pc[:, 1]
conn_pc1 = conn_pc[:, 0]

# Path a: Conn → GenePC2
rho_a, p_a = spearmanr(conn_pc1, gene_pc2)
rho_a_p, p_a_p = partial_spearman(conn_pc1, gene_pc2, layer_vec)
print(f"  Path a (ConnPC1 → GenePC2): ρ={rho_a:+.4f} p={p_a:.4f} | partial: ρ={rho_a_p:+.4f} p={p_a_p:.4f}")

# Path b: GenePC2 → Burst (controlling for ConnPC1)
from scipy.stats import rankdata
def double_partial(x, y, z1, z2):
    rx = rankdata(x)
    ry = rankdata(y)
    rz1 = rankdata(z1)
    rz2 = rankdata(z2)
    X = np.column_stack([np.ones(len(rx)), rz1, rz2])
    beta_x = np.linalg.lstsq(X, rx, rcond=None)[0]
    res_x = rx - X @ beta_x
    beta_y = np.linalg.lstsq(X, ry, rcond=None)[0]
    res_y = ry - X @ beta_y
    r, p = pearsonr(res_x, res_y)
    return float(r), float(p)

# Path c: ConnPC1 → Burst (total effect)
rho_c, p_c = spearmanr(conn_pc1, burst_vals)
rho_c_p, p_c_p = partial_spearman(conn_pc1, burst_vals, layer_vec)
print(f"  Path c (ConnPC1 → Burst, total): ρ={rho_c:+.4f} p={p_c:.4f} | partial: ρ={rho_c_p:+.4f} p={p_c_p:.4f}")

# Path c': ConnPC1 → Burst (controlling for GenePC2, direct effect)
rho_cp, p_cp = double_partial(conn_pc1, burst_vals, gene_pc2, layer_vec)
print(f"  Path c' (ConnPC1 → Burst, ctrl GenePC2+layer): ρ={rho_cp:+.4f} p={p_cp:.4f}")

# Path b: GenePC2 → Burst (controlling for ConnPC1 + layer)
rho_b, p_b = double_partial(gene_pc2, burst_vals, conn_pc1, layer_vec)
print(f"  Path b (GenePC2 → Burst, ctrl ConnPC1+layer): ρ={rho_b:+.4f} p={p_b:.4f}")

# Indirect effect: a × b
indirect_raw = rho_a * rho_b if rho_a is not None and rho_b is not None else None
print(f"  Indirect effect (a×b, raw): {indirect_raw:.4f}" if indirect_raw is not None else "  Indirect effect: N/A")

# Sobel-like test (approximate)
if indirect_raw is not None:
    from scipy.stats import norm
    se_a = 1.0 / np.sqrt(10 - 3)
    se_b = 1.0 / np.sqrt(10 - 4)
    se_indirect = np.sqrt(rho_b**2 * se_a**2 + rho_a**2 * se_b**2)
    z_sobel = indirect_raw / se_indirect
    p_sobel = 2 * (1 - norm.cdf(abs(z_sobel)))
    print(f"  Sobel test: z={z_sobel:.4f}, p={p_sobel:.4f}")

print("\n[7] Permutation test for mediation...")

# Bootstrap mediation test
n_boot = 10000
indirect_boots = []
for _ in range(n_boot):
    idx = np.random.choice(len(ALL_REGIONS), size=len(ALL_REGIONS), replace=True)
    try:
        a_boot, _ = spearmanr(conn_pc1[idx], gene_pc2[idx])
        # GenePC2 → Burst controlling for ConnPC1
        rx = rankdata(gene_pc2[idx])
        ry = rankdata(burst_vals[idx])
        rz = rankdata(conn_pc1[idx])
        X = np.column_stack([np.ones(len(rx)), rz])
        beta_x = np.linalg.lstsq(X, rx, rcond=None)[0]
        res_x = rx - X @ beta_x
        beta_y = np.linalg.lstsq(X, ry, rcond=None)[0]
        res_y = ry - X @ beta_y
        b_boot, _ = pearsonr(res_x, res_y)
        indirect_boots.append(a_boot * b_boot)
    except Exception:
        continue

indirect_boots = np.array(indirect_boots)
ci_lo = np.percentile(indirect_boots, 2.5)
ci_hi = np.percentile(indirect_boots, 97.5)
p_mediation = (np.sum(indirect_boots <= 0) + 1) / (n_boot + 1) if indirect_raw > 0 else (np.sum(indirect_boots >= 0) + 1) / (n_boot + 1)
print(f"  Indirect effect bootstrap: mean={indirect_boots.mean():.4f}")
print(f"  95% CI: [{ci_lo:.4f}, {ci_hi:.4f}]")
print(f"  Mediation p (one-tailed): {p_mediation:.4f}")
if ci_lo > 0 or ci_hi < 0:
    print(f"  CI does NOT include 0 → SIGNIFICANT mediation")
else:
    print(f"  CI includes 0 → non-significant mediation")

print("\n[8] Testing all connectivity PCs as mediators...")

mediation_results = []
for ci in range(min(5, conn_pc.shape[1])):
    for gi in range(min(5, gene_pc.shape[1])):
        conn_pc_i = conn_pc[:, ci]
        gene_pc_i = gene_pc[:, gi]

        rho_a_i, p_a_i = spearmanr(conn_pc_i, gene_pc_i)
        if p_a_i > 0.2:
            continue

        rho_c_i, p_c_i = spearmanr(conn_pc_i, burst_vals)
        rho_cp_i, p_cp_i = double_partial(conn_pc_i, burst_vals, gene_pc_i, layer_vec)
        rho_b_i, p_b_i = double_partial(gene_pc_i, burst_vals, conn_pc_i, layer_vec)

        indirect_i = rho_a_i * rho_b_i

        mediation_results.append({
            "conn_pc": ci + 1,
            "gene_pc": gi + 1,
            "a_path": round(rho_a_i, 4),
            "a_p": round(p_a_i, 6),
            "b_path": round(rho_b_i, 4),
            "b_p": round(p_b_i, 6),
            "c_total": round(rho_c_i, 4),
            "c_p": round(p_c_i, 6),
            "c_direct": round(rho_cp_i, 4),
            "c_direct_p": round(p_cp_i, 6),
            "indirect": round(indirect_i, 4),
        })

mediation_results.sort(key=lambda x: abs(x["indirect"]), reverse=True)
print(f"  Pathways with a_path p<0.2: {len(mediation_results)}")
for mr in mediation_results[:10]:
    print(f"    ConnPC{mr['conn_pc']} → GenePC{mr['gene_pc']} → Burst: "
          f"a={mr['a_path']:+.3f}(p={mr['a_p']:.3f}), "
          f"b={mr['b_path']:+.3f}(p={mr['b_p']:.3f}), "
          f"c={mr['c_total']:+.3f}(p={mr['c_p']:.3f}), "
          f"c'={mr['c_direct']:+.3f}(p={mr['c_direct_p']:.3f}), "
          f"indirect={mr['indirect']:+.3f}")

mediation_sig = any(mr["c_direct_p"] > 0.05 and mr["c_p"] < 0.1 and mr["indirect"] != 0
                    for mr in mediation_results)

partial_mediation = any(mr["c_direct_p"] > mr["c_p"] and abs(mr["c_direct"]) < abs(mr["c_total"])
                        for mr in mediation_results)

status = "CONFIRMED" if (p_mediation < 0.05 and partial_mediation) else "REFUTED"

results = {
    "hypothesis_id": "H29",
    "status": status,
    "primary_mediation": {
        "conn_pc": 1, "gene_pc": 2,
        "a_path": round(rho_a, 4), "a_p": round(p_a, 6),
        "a_partial": round(rho_a_p, 4), "a_partial_p": round(p_a_p, 6),
        "b_path": round(rho_b, 4), "b_p": round(p_b, 6),
        "c_total": round(rho_c, 4), "c_p": round(p_c, 6),
        "c_total_partial": round(rho_c_p, 4), "c_total_partial_p": round(p_c_p, 6),
        "c_direct": round(rho_cp, 4), "c_direct_p": round(p_cp, 6),
        "indirect": round(indirect_raw, 4) if indirect_raw is not None else None,
        "indirect_ci": [round(ci_lo, 4), round(ci_hi, 4)],
        "mediation_p": round(p_mediation, 6),
    },
    "all_mediation_paths": mediation_results[:15],
    "conn_pca_variance": [round(float(v), 4) for v in pca_conn.explained_variance_ratio_[:5]],
    "gene_pca_variance": [round(float(v), 4) for v in pca_gene.explained_variance_ratio_[:5]],
    "notes": f"Primary: ConnPC1→GenePC2→Burst indirect={indirect_raw:.3f} p={p_mediation:.4f}; {status}"
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
ax.scatter(conn_pc1, gene_pc2, c=["#4C72B0" if r in L5_REGIONS else "#DD8452" for r in ALL_REGIONS],
           s=100, alpha=0.8, edgecolors="black")
for i, r in enumerate(ALL_REGIONS):
    ax.annotate(r, (conn_pc1[i], gene_pc2[i]), fontsize=7, ha="left", va="bottom")
ax.set_xlabel("ConnPC1")
ax.set_ylabel("GenePC2 (burst module)")
ax.set_title(f"Path a: Conn → Gene\nρ={rho_a:+.3f}, p={p_a:.3f}")
from matplotlib.patches import Patch
ax.legend(handles=[Patch(facecolor="#4C72B0", label="L5"), Patch(facecolor="#DD8452", label="L6")], fontsize=8)

ax = axes[1]
ax.scatter(gene_pc2, burst_vals, c=["#4C72B0" if r in L5_REGIONS else "#DD8452" for r in ALL_REGIONS],
           s=100, alpha=0.8, edgecolors="black")
for i, r in enumerate(ALL_REGIONS):
    ax.annotate(r, (gene_pc2[i], burst_vals[i]), fontsize=7, ha="left", va="bottom")
ax.set_xlabel("GenePC2 (burst module)")
ax.set_ylabel("Burst index")
ax.set_title(f"Path b: Gene → Burst\nρ_partial={rho_b:+.3f}, p={p_b:.3f}")

ax = axes[2]
ax.scatter(conn_pc1, burst_vals, c=["#4C72B0" if r in L5_REGIONS else "#DD8452" for r in ALL_REGIONS],
           s=100, alpha=0.8, edgecolors="black")
for i, r in enumerate(ALL_REGIONS):
    ax.annotate(r, (conn_pc1[i], burst_vals[i]), fontsize=7, ha="left", va="bottom")
ax.set_xlabel("ConnPC1")
ax.set_ylabel("Burst index")
ax.set_title(f"Path c: Conn → Burst\nρ={rho_c:+.3f}(total), ρ'={rho_cp:+.3f}(direct)")

plt.suptitle("H29: Connectivity → Gene expression → Burst index chain", fontsize=14)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "figures" / "connectivity_gene_ephys_chain.png", dpi=150, bbox_inches="tight")
plt.savefig(OUTPUT_DIR / "figures" / "connectivity_gene_ephys_chain.svg", bbox_inches="tight")

print(f"\nResults saved to {out_path}")
print("Done.")

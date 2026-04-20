"""
Hypothesis H02: PFC neurons with the same transcriptomic class (glutamatergic vs GABAergic)
show similar electrophysiological subtype distributions within the same cortical region.
Branch: transcr-electro-pfc
Datasets: neural_activity, merfish
"""

import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from scipy.stats import chi2_contingency, fisher_exact, pearsonr, spearmanr

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

DATA_PATH = Path(__file__).parent.parent.parent / "data"
OUTPUT_DIR = Path(__file__).parent


def load_neural_activity() -> pl.DataFrame:
    df = pl.read_csv(DATA_PATH / "pfcmap" / "units.csv")
    pfc_regions = ["ORBvl", "ORBl", "ORBm", "ILA", "PL"]
    df = df.filter(
        pl.col("region").str.contains("|".join(pfc_regions))
    )
    df = df.filter(pl.col("utype").is_in(["ww", "nw"]))
    df = df.filter(pl.col("layer") != "NA")
    return df


def load_merfish() -> pl.DataFrame:
    df = pl.read_csv(DATA_PATH / "merfish" / "cell_metadata_with_parcellation_annotation.csv")
    pfc_regions = ["ORBvl", "ORBl", "ORBm", "ILA", "PL"]
    df = df.filter(
        df["parcellation_substructure"].str.contains("|".join(pfc_regions))
    )
    return df


def compute_ww_fraction_per_region(df: pl.DataFrame) -> pl.DataFrame:
    return df.group_by("region").agg([
        pl.col("utype").filter(pl.col("utype") == "ww").count().alias("n_ww"),
        pl.col("utype").count().alias("n_total")
    ]).with_columns(
        (pl.col("n_ww") / pl.col("n_total")).alias("ww_fraction")
    )


def compute_glut_fraction_per_region(df: pl.DataFrame) -> pl.DataFrame:
    return df.group_by("parcellation_substructure").agg([
        pl.col("class").filter(pl.col("class").str.contains("Glut")).count().alias("n_glut"),
        pl.col("class").count().alias("n_total")
    ]).with_columns(
        (pl.col("n_glut") / pl.col("n_total")).alias("glut_fraction")
    )


def permutation_test_correlation(x: np.ndarray, y: np.ndarray, n_permutations: int = 10000) -> float:
    observed_r, _ = pearsonr(x, y)
    null_r = []
    for _ in range(n_permutations):
        y_perm = np.random.permutation(y)
        null_r.append(pearsonr(x, y_perm)[0])
    p_value = np.mean(np.abs(null_r) >= np.abs(observed_r))
    return p_value, observed_r


print("Loading datasets...")
na = load_neural_activity()
mer = load_merfish()

print(f"Neural activity: {len(na)} neurons from {na['region'].n_unique()} PFC regions")
print(f"MERFISH: {len(mer)} cells from {mer['parcellation_substructure'].n_unique()} PFC regions")

region_ww = compute_ww_fraction_per_region(na)
region_glut = compute_glut_fraction_per_region(mer)

merged = region_ww.join(
    region_glut,
    left_on="region",
    right_on="parcellation_substructure",
    how="inner"
)

min_cells = 50
merged = merged.filter(
    (pl.col("n_total") >= min_cells) & (pl.col("n_total_right") >= min_cells)
)

ww_frac = merged["ww_fraction"].to_numpy()
glut_frac = merged["glut_fraction"].to_numpy()
n_regions = len(ww_frac)

print(f"\nAfter filtering (>= {min_cells} cells per region): {n_regions} regions")

r, p_pearson = pearsonr(ww_frac, glut_frac)
rho, p_spearman = spearmanr(ww_frac, glut_frac)

print(f"\n=== Correlation Analysis ===")
print(f"Pearson r = {r:.3f}, p = {p_pearson:.4f}")
print(f"Spearman rho = {rho:.3f}, p = {p_spearman:.4f}")

p_perm, r_perm = permutation_test_correlation(ww_frac, glut_frac, n_permutations=10000)
print(f"Permutation test p = {p_perm:.4f}")

status = "REFUTED"
if p_pearson < 0.05 and r > 0:
    status = "CONFIRMED"

results = {
    "hypothesis_id": "H02",
    "status": status,
    "statistic": float(r),
    "p_value": float(p_pearson),
    "effect_size": float(r),
    "n_cells": int(len(na) + len(mer)),
    "n_regions": int(n_regions),
    "notes": f"Pearson r = {r:.3f}, p = {p_pearson:.4f}; Spearman rho = {rho:.3f}"
}

print("\n" + json.dumps(results, indent=2))

out_path = OUTPUT_DIR / "results.json"
out_path.write_text(json.dumps(results, indent=2))

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].scatter(ww_frac, glut_frac, s=100, alpha=0.7)
axes[0].set_xlabel("Wide-width neuron fraction (electrophysiology)")
axes[0].set_ylabel("Glutamatergic neuron fraction (MERFISH)")
axes[0].set_title(f"Cross-modal Correlation in PFC Regions\nr = {r:.3f}, p = {p_pearson:.4f}")

z = np.polyfit(ww_frac, glut_frac, 1)
p = np.poly1d(z)
x_line = np.linspace(ww_frac.min(), ww_frac.max(), 100)
axes[0].plot(x_line, p(x_line), "r--", alpha=0.8, label="Linear fit")
axes[0].legend()

for region, ww_f, glut_f in merged.select(['region', 'ww_fraction', 'glut_fraction']).iter_rows():
    axes[0].annotate(region[:6], (ww_f, glut_f), fontsize=8)

region_summary = merged.select(["region", "ww_fraction", "glut_fraction", "n_total", "n_total_right"])
region_summary = region_summary.sort("ww_fraction", descending=True)
regions = region_summary["region"].to_list()
effect_sizes = (region_summary["glut_fraction"] - region_summary["glut_fraction"].mean()).to_list()

axes[1].barh(regions, effect_sizes)
axes[1].axvline(0, color="black", linestyle="--")
axes[1].set_xlabel("Deviation from mean glutamatergic fraction")
axes[1].set_title("Glutamate Fraction by Region")

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "figures" / "transcr_electro_correlation.png", dpi=150)
plt.savefig(OUTPUT_DIR / "figures" / "transcr_electro_correlation.svg")

print(f"\nResults saved to {out_path}")
print("Figures saved to figures/")
"""
Hypothesis H05: The Sst-to-Pvalb interneuron ratio in PFC regions is positively
correlated with the proportion of narrow-width (fast-spiking) electrophysiological
neurons across those same regions.
Branch: sst-pv-nw-ratio
Datasets: merfish, neural_activity
"""

import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from scipy.stats import pearsonr, spearmanr

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

DATA_PATH = Path(__file__).parent.parent.parent / "data"
OUTPUT_DIR = Path(__file__).parent

PFC_REGIONS = [
    "PL5", "PL2/3", "PL6a",
    "ILA5", "ILA6a",
    "ORBvl5", "ORBl5", "ORBm5", "ORBm6a",
    "ORBl2/3", "ORBvl2/3",
    "ORBl6a", "ORBvl6a",
]


def load_merfish() -> pl.DataFrame:
    df = pl.read_csv(
        DATA_PATH / "merfish" / "cell_metadata_with_parcellation_annotation.csv"
    )
    pfc_pat = "|".join(PFC_REGIONS)
    df = df.filter(
        df["parcellation_substructure"].str.contains(pfc_pat)
    )
    return df


def load_neural_activity() -> pl.DataFrame:
    df = pl.read_csv(DATA_PATH / "pfcmap" / "units.csv")
    pfc_areas = ["ORBvl", "ORBl", "ORBm", "ILA", "PL"]
    df = df.filter(
        pl.col("region").str.contains("|".join(pfc_areas))
    )
    df = df.filter(pl.col("utype").is_in(["ww", "nw"]))
    return df


def compute_sst_pv_ratio(merfish: pl.DataFrame) -> dict:
    ratios = {}
    for r in PFC_REGIONS:
        sub = merfish.filter(pl.col("parcellation_substructure") == r)
        if len(sub) < 100:
            continue
        sst = (sub["subclass"] == "053 Sst Gaba").sum()
        pv = (sub["subclass"] == "052 Pvalb Gaba").sum()
        if pv > 0:
            ratios[r] = float(sst) / float(pv)
    return ratios


def compute_nw_fraction(neural_activity: pl.DataFrame) -> pl.DataFrame:
    return neural_activity.group_by("region").agg([
        (pl.col("utype") == "nw").sum().alias("n_nw"),
        pl.len().alias("n_total"),
    ]).with_columns(
        (pl.col("n_nw") / pl.col("n_total")).alias("nw_fraction")
    ).filter(pl.col("n_total") >= 50)


def permutation_correlation(x: np.ndarray, y: np.ndarray, n_perm: int = 10000) -> float:
    observed_r, _ = pearsonr(x, y)
    null_r = []
    for _ in range(n_perm):
        y_perm = np.random.permutation(y)
        null_r.append(pearsonr(x, y_perm)[0])
    null_r = np.array(null_r)
    p_value = float(np.mean(np.abs(null_r) >= np.abs(observed_r)))
    return p_value


print("Loading datasets...")
merfish = load_merfish()
neural_activity = load_neural_activity()

print(f"MERFISH: {len(merfish)} cells")
print(f"Neural activity: {len(neural_activity)} neurons")

sst_pv = compute_sst_pv_ratio(merfish)
nw_frac_df = compute_nw_fraction(neural_activity)

regions = []
ratios = []
nw_fracs = []
for r in PFC_REGIONS:
    if r in sst_pv:
        row = nw_frac_df.filter(pl.col("region") == r)
        if len(row) > 0:
            regions.append(r)
            ratios.append(sst_pv[r])
            nw_fracs.append(row["nw_fraction"].item())

ratios = np.array(ratios)
nw_fracs = np.array(nw_fracs)
n_regions = len(regions)

print(f"\nOverlapping regions with sufficient data: {n_regions}")

if n_regions < 5:
    results = {
        "hypothesis_id": "H05",
        "status": "INCONCLUSIVE",
        "statistic": None,
        "p_value": None,
        "effect_size": None,
        "n_cells": len(merfish) + len(neural_activity),
        "notes": f"Only {n_regions} regions with sufficient data (need >= 5)"
    }
else:
    r_pearson, p_pearson = pearsonr(ratios, nw_fracs)
    rho_spearman, p_spearman = spearmanr(ratios, nw_fracs)
    p_perm = permutation_correlation(ratios, nw_fracs, n_perm=10000)

    print(f"\n=== Correlation Analysis ===")
    print(f"Pearson r = {r_pearson:.3f}, p = {p_pearson:.4f}")
    print(f"Spearman rho = {rho_spearman:.3f}, p = {p_spearman:.4f}")
    print(f"Permutation p = {p_perm:.4f}")

    for i, reg in enumerate(regions):
        print(f"  {reg}: SST/PV={ratios[i]:.2f}, NW%={nw_fracs[i]*100:.1f}")

    status = "CONFIRMED" if (p_pearson < 0.05 or p_spearman < 0.05) and r_pearson > 0 else "REFUTED"

    results = {
        "hypothesis_id": "H05",
        "status": status,
        "statistic": float(r_pearson),
        "p_value": float(p_pearson),
        "effect_size": float(r_pearson),
        "n_cells": int(len(merfish) + len(neural_activity)),
        "n_regions": int(n_regions),
        "pearson_r": float(r_pearson),
        "pearson_p": float(p_pearson),
        "spearman_rho": float(rho_spearman),
        "spearman_p": float(p_spearman),
        "permutation_p": float(p_perm),
        "notes": f"Pearson r={r_pearson:.3f}, p={p_pearson:.4f}; Spearman rho={rho_spearman:.3f}, p={p_spearman:.4f}"
    }

print("\n" + json.dumps(results, indent=2))

out_path = OUTPUT_DIR / "results.json"
out_path.write_text(json.dumps(results, indent=2))

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].scatter(ratios, nw_fracs, s=100, alpha=0.7, edgecolors="black")
for i, reg in enumerate(regions):
    axes[0].annotate(reg[:6], (ratios[i], nw_fracs[i]), fontsize=7, xytext=(5, 5), textcoords="offset points")
z = np.polyfit(ratios, nw_fracs, 1)
p_line = np.poly1d(z)
x_line = np.linspace(ratios.min(), ratios.max(), 100)
axes[0].plot(x_line, p_line(x_line), "r--", alpha=0.8)
axes[0].set_xlabel("Sst/Pvalb ratio (MERFISH)")
axes[0].set_ylabel("Narrow-width neuron fraction (electrophysiology)")
axes[0].set_title(f"SST/PV Ratio vs NW Fraction\nr = {results.get('pearson_r', 0):.3f}, p = {results.get('pearson_p', 1):.4f}")

pv_frac = []
sst_frac = []
for r in regions:
    sub = merfish.filter(pl.col("parcellation_substructure") == r)
    pv_frac.append(float((sub["subclass"] == "052 Pvalb Gaba").sum()) / len(sub))
    sst_frac.append(float((sub["subclass"] == "053 Sst Gaba").sum()) / len(sub))

x_pos = np.arange(len(regions))
width = 0.35
axes[1].bar(x_pos - width / 2, [s * 100 for s in pv_frac], width, label="Pvalb %", color="#4C72B0")
axes[1].bar(x_pos + width / 2, [s * 100 for s in sst_frac], width, label="Sst %", color="#DD8452")
axes[1].set_xticks(x_pos)
axes[1].set_xticklabels(regions, rotation=45, ha="right", fontsize=7)
axes[1].set_ylabel("Fraction of total cells (%)")
axes[1].set_title("PV and SST Composition by Region")
axes[1].legend()

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "figures" / "sst_pv_nw_ratio.png", dpi=150)
plt.savefig(OUTPUT_DIR / "figures" / "sst_pv_nw_ratio.svg")

print(f"\nResults saved to {out_path}")
print("Figures saved to figures/")

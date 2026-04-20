"""
Hypothesis H06: Narrow-width (fast-spiking) PFC neurons have higher burst
indices than wide-width (regular-spiking) PFC neurons, reflecting burst-like
firing patterns in interneurons.
Branch: nw-burst-index
Datasets: neural_activity
"""

import json
import random
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from scipy.stats import mannwhitneyu, wilcoxon

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

DATA_PATH = Path(__file__).parent.parent.parent / "data"
OUTPUT_DIR = Path(__file__).parent

PFC_AREAS = ["ORBvl", "ORBl", "ORBm", "ILA", "PL"]


def load_neural_activity() -> pl.DataFrame:
    df = pl.read_csv(DATA_PATH / "pfcmap" / "units.csv")
    df = df.filter(
        pl.col("region").str.contains("|".join(PFC_AREAS))
    )
    df = df.filter(pl.col("utype").is_in(["ww", "nw"]))
    df = df.filter(pl.col("B_mean").is_not_null())
    return df


def permutation_paired_test(
    nw_means: np.ndarray, ww_means: np.ndarray, n_perm: int = 10000
) -> float:
    observed_diff = np.mean(nw_means) - np.mean(ww_means)
    all_means = np.concatenate([nw_means, ww_means])
    n_nw = len(nw_means)
    null_diffs = []
    for _ in range(n_perm):
        perm = np.random.permutation(all_means)
        null_diffs.append(perm[:n_nw].mean() - perm[n_nw:].mean())
    null_diffs = np.array(null_diffs)
    p_value = float(np.mean(null_diffs >= observed_diff))
    return p_value


print("Loading neural activity data...")
df = load_neural_activity()
print(f"Loaded {len(df)} neurons")

nw = df.filter(pl.col("utype") == "nw")
ww = df.filter(pl.col("utype") == "ww")

nw_burst = nw["B_mean"].to_numpy()
ww_burst = ww["B_mean"].to_numpy()

print(f"\nNarrow-width: n={len(nw_burst)}, mean B_mean={nw_burst.mean():.4f}")
print(f"Wide-width: n={len(ww_burst)}, mean B_mean={ww_burst.mean():.4f}")
print(f"Difference: {nw_burst.mean() - ww_burst.mean():.4f}")

stat_global, p_global = mannwhitneyu(nw_burst, ww_burst, alternative="greater")
print(f"\nCell-level Mann-Whitney U (NW > WW): {stat_global:.0f}, p={p_global:.2e}")

# Region-level paired test
regions = df["region"].unique().to_list()
nw_region_means = []
ww_region_means = []
region_names = []

for r in regions:
    sub = df.filter(pl.col("region") == r)
    nw_r = sub.filter(pl.col("utype") == "nw")
    ww_r = sub.filter(pl.col("utype") == "ww")
    if len(nw_r) >= 5 and len(ww_r) >= 5:
        nw_region_means.append(nw_r["B_mean"].mean())
        ww_region_means.append(ww_r["B_mean"].mean())
        region_names.append(r)

nw_region_means = np.array(nw_region_means)
ww_region_means = np.array(ww_region_means)

print(f"\n=== Region-level paired test ({len(region_names)} regions) ===")
for i, r in enumerate(region_names):
    print(f"  {r}: NW={nw_region_means[i]:.4f}, WW={ww_region_means[i]:.4f}, diff={nw_region_means[i]-ww_region_means[i]:.4f}")

stat_paired, p_paired = wilcoxon(nw_region_means, ww_region_means, alternative="greater")
print(f"\nWilcoxon signed-rank (NW > WW B_mean): stat={stat_paired:.1f}, p={p_paired:.2e}")

p_perm = permutation_paired_test(nw_region_means, ww_region_means, n_perm=10000)
print(f"Permutation test p: {p_perm:.4f}")

n_regions_with_higher_nw = int(np.sum(nw_region_means > ww_region_means))
print(f"Regions where NW > WW: {n_regions_with_higher_nw}/{len(region_names)}")

status = "CONFIRMED" if p_paired < 0.05 else "REFUTED"

results = {
    "hypothesis_id": "H06",
    "status": status,
    "statistic": float(stat_paired),
    "p_value": float(p_paired),
    "effect_size": float(nw_burst.mean() - ww_burst.mean()),
    "n_cells": int(len(df)),
    "n_nw": int(len(nw_burst)),
    "n_ww": int(len(ww_burst)),
    "n_regions": int(len(region_names)),
    "nw_mean_burst": float(nw_burst.mean()),
    "ww_mean_burst": float(ww_burst.mean()),
    "cell_level_p": float(p_global),
    "permutation_p": float(p_perm),
    "notes": f"NW B_mean ({nw_burst.mean():.4f}) > WW B_mean ({ww_burst.mean():.4f}), p_paired={p_paired:.2e}"
}

print("\n" + json.dumps(results, indent=2))

out_path = OUTPUT_DIR / "results.json"
out_path.write_text(json.dumps(results, indent=2))

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].boxplot(
    [nw_burst, ww_burst],
    tick_labels=["Narrow-width\n(fast-spiking)", "Wide-width\n(regular-spiking)"],
    patch_artist=True,
)
axes[0].boxes = axes[0].findobj(lambda x: hasattr(x, 'set_facecolor'))
for i, box in enumerate(axes[0].findobj(matplotlib.patches.PathPatch)):
    box.set_facecolor(["#DD8452", "#4C72B0"][i])
    box.set_alpha(0.7)
axes[0].set_ylabel("Burst index (B_mean)")
axes[0].set_title(f"Burst Index by Subtype\np = {p_global:.2e}")

x_pos = np.arange(len(region_names))
width = 0.35
axes[1].bar(x_pos - width / 2, nw_region_means, width, label="NW", color="#DD8452", edgecolor="black")
axes[1].bar(x_pos + width / 2, ww_region_means, width, label="WW", color="#4C72B0", edgecolor="black")
axes[1].set_xticks(x_pos)
axes[1].set_xticklabels(region_names, rotation=45, ha="right", fontsize=7)
axes[1].set_ylabel("Mean burst index")
axes[1].set_title("Burst Index by Region and Subtype")
axes[1].legend()
axes[1].axhline(0, color="black", linestyle="--", alpha=0.5)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "figures" / "nw_burst_index.png", dpi=150)
plt.savefig(OUTPUT_DIR / "figures" / "nw_burst_index.svg")

print(f"\nResults saved to {out_path}")
print("Figures saved to figures/")

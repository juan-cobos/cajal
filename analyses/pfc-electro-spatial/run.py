"""
Hypothesis H01: PFC neurons with the same electrophysiological subtype (wide-width vs narrow-width)
are spatially clustered closer together than neurons with different subtypes within the same cortical region.
Branch: pfc-electro-spatial
Datasets: neural_activity
"""

import json
import random
from pathlib import Path

import numpy as np
import polars as pl
from scipy.spatial.distance import cdist
from scipy.stats import mannwhitneyu

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

DATA_PATH = Path(__file__).parent.parent.parent / "data"
OUTPUT_DIR = Path(__file__).parent


def load_neural_activity() -> pl.DataFrame:
    """Load neural activity data, filtering to PFC-related regions with utype."""
    df = pl.read_csv(DATA_PATH / "pfcmap" / "units.csv")

    pfc_regions = [
        "ORBvl", "ORBl", "ORBm", "ILA", "PL"
    ]
    df = df.filter(
        pl.col("region").str.contains("|".join(pfc_regions))
    )
    df = df.filter(pl.col("utype").is_in(["ww", "nw"]))
    df = df.filter(pl.col("layer") != "NA")

    return df


def compute_pairwise_distances(coords: np.ndarray) -> np.ndarray:
    """Compute pairwise Euclidean distances between all points."""
    return cdist(coords, coords, metric="euclidean")


def get_within_group_distances(dist_matrix: np.ndarray, labels: np.ndarray, group: str) -> np.ndarray:
    """Get distances between pairs of same group."""
    idx = np.where(labels == group)[0]
    dists = []
    for i in range(len(idx)):
        for j in range(i + 1, len(idx)):
            dists.append(dist_matrix[idx[i], idx[j]])
    return np.array(dists)


def get_between_group_distances(dist_matrix: np.ndarray, labels: np.ndarray, group1: str, group2: str) -> np.ndarray:
    """Get distances between pairs of different groups."""
    idx1 = np.where(labels == group1)[0]
    idx2 = np.where(labels == group2)[0]
    dists = []
    for i in idx1:
        for j in idx2:
            dists.append(dist_matrix[i, j])
    return np.array(dists)


def permutation_test(observed_diff: float, n_permutations: int = 1000) -> float:
    """Permutation test: randomly shuffle labels and compute null distribution."""
    null_diffs = []
    for _ in range(n_permutations):
        shuffled = np.random.permutation(labels)
        within_ww = get_within_group_distances(dist_matrix, shuffled, "ww")
        within_nw = get_within_group_distances(dist_matrix, shuffled, "nw")
        if len(within_ww) > 0 and len(within_nw) > 0:
            null_diffs.append(np.mean(within_ww) - np.mean(within_nw))
    if len(null_diffs) == 0:
        return 1.0
    null_diffs = np.array(null_diffs)
    p_value = np.mean(np.abs(null_diffs) >= np.abs(observed_diff))
    return p_value


df = load_neural_activity()
print(f"Loaded {len(df)} neurons from PFC regions")
print(f"Unique regions: {df['region'].n_unique()}")
print(f"UTypes: {df['utype'].value_counts().to_dict()}")

region_results = []
all_within_same = []
all_within_diff = []
all_between = []

for region in df["region"].unique():
    region_df = df.filter(pl.col("region") == region)

    if len(region_df) < 10:
        continue

    utypes = region_df["utype"].unique().to_list()
    if "ww" not in utypes or "nw" not in utypes:
        continue

    coords = region_df.select(["AP", "DV", "ML"]).to_numpy()
    labels = np.array(region_df["utype"].to_list())

    dist_matrix = compute_pairwise_distances(coords)

    within_ww = get_within_group_distances(dist_matrix, labels, "ww")
    within_nw = get_within_group_distances(dist_matrix, labels, "nw")
    between = get_between_group_distances(dist_matrix, labels, "ww", "nw")

    if len(within_ww) < 5 or len(within_nw) < 5 or len(between) < 5:
        continue

    mean_within = (np.mean(within_ww) + np.mean(within_nw)) / 2
    mean_between = np.mean(between)

    region_results.append({
        "region": region,
        "n_cells": len(region_df),
        "n_ww": np.sum(labels == "ww"),
        "n_nw": np.sum(labels == "nw"),
        "mean_within_group_distance": float(mean_within),
        "mean_between_group_distance": float(mean_between),
        "effect_size": float(mean_between - mean_within)
    })

    all_within_same.extend(within_ww.tolist())
    all_within_same.extend(within_nw.tolist())
    all_between.extend(between.tolist())

if len(all_within_same) < 10 or len(all_between) < 10:
    print("Insufficient data for global analysis")
    results = {
        "hypothesis_id": "H01",
        "status": "INCONCLUSIVE",
        "statistic": None,
        "p_value": None,
        "effect_size": None,
        "n_cells": len(df),
        "notes": "Insufficient neurons with both ww and nw in same regions"
    }
else:
    all_within_same = np.array(all_within_same)
    all_between = np.array(all_between)

    stat, p_value = mannwhitneyu(all_within_same, all_between, alternative="less")
    effect_size = np.mean(all_between) - np.mean(all_within_same)

    print(f"\n=== Global Results ===")
    print(f"N within-group pairs: {len(all_within_same)}")
    print(f"N between-group pairs: {len(all_between)}")
    print(f"Mean within-group distance: {np.mean(all_within_same):.2f}")
    print(f"Mean between-group distance: {np.mean(all_between):.2f}")
    print(f"Effect size (between - within): {effect_size:.2f}")
    print(f"Mann-Whitney U statistic: {stat:.2f}")
    print(f"P-value (one-sided, within < between): {p_value:.6f}")

    status = "CONFIRMED" if p_value < 0.05 and effect_size > 0 else "REFUTED"

    results = {
        "hypothesis_id": "H01",
        "status": status,
        "statistic": float(stat),
        "p_value": float(p_value),
        "effect_size": float(effect_size),
        "n_cells": int(len(df)),
        "notes": f"Within-group distance < between-group distance: {status}"
    }

print("\n" + json.dumps(results, indent=2))

out_path = OUTPUT_DIR / "results.json"
out_path.write_text(json.dumps(results, indent=2))

import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].hist(all_within_same, bins=50, alpha=0.7, label="Within same subtype", density=True)
axes[0].hist(all_between, bins=50, alpha=0.7, label="Between subtypes", density=True)
axes[0].set_xlabel("Pairwise distance (μm)")
axes[0].set_ylabel("Density")
axes[0].set_title("Spatial Clustering of Electrophysiological Subtypes")
axes[0].legend()

region_names = [r["region"] for r in region_results]
effect_sizes = [r["effect_size"] for r in region_results]
axes[1].barh(region_names, effect_sizes)
axes[1].axvline(0, color="black", linestyle="--")
axes[1].set_xlabel("Effect size (between - within group distance)")
axes[1].set_title("Effect Size by Region")

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "figures" / "spatial_clustering.png", dpi=150)
plt.savefig(OUTPUT_DIR / "figures" / "spatial_clustering.svg")

print(f"\nResults saved to {out_path}")
print("Figures saved to figures/")
"""
Hypothesis H07: PFC regions with higher transcriptomic (subclass) diversity
have lower mean firing rates — more transcriptomically heterogeneous regions
fire more slowly.

Branch: merfish-entropy-vs-rate
Datasets: merfish, neural_activity
"""

import json
import random
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from scipy.stats import spearmanr

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

DATA_PATH = Path(__file__).parent.parent.parent / "data"
OUTPUT_DIR = Path(__file__).parent

SUBSTRUCTURE_TO_PARENT = {
    "PL5": "PL", "PL2/3": "PL", "PL6a": "PL", "PL1": "PL",
    "ILA5": "ILA", "ILA6a": "ILA", "ILA2/3": "ILA", "ILA1": "ILA",
    "ORBl5": "ORBl", "ORBl2/3": "ORBl", "ORBl6a": "ORBl", "ORBl1": "ORBl",
    "ORBm5": "ORBm", "ORBm2/3": "ORBm", "ORBm6a": "ORBm", "ORBm1": "ORBm",
    "ORBvl5": "ORBvl", "ORBvl2/3": "ORBvl", "ORBvl6a": "ORBvl", "ORBvl1": "ORBvl",
    "ACAd5": "ACAd", "ACAd2/3": "ACAd", "ACAd6a": "ACAd", "ACAd1": "ACAd",
    "ACAv5": "ACAv", "ACAv2/3": "ACAv", "ACAv6a": "ACAv", "ACAv1": "ACAv",
    "FRP5": "FRP", "FRP2/3": "FRP", "FRP6a": "FRP", "FRP1": "FRP",
}


def shannon_entropy(counts: Counter) -> float:
    total = sum(counts.values())
    if total == 0:
        return 0.0
    probs = np.array([c / total for c in counts.values() if c > 0])
    return float(-np.sum(probs * np.log2(probs)))


def load_merfish_pfc() -> pl.DataFrame:
    df = pl.read_csv(
        DATA_PATH / "merfish" / "cell_metadata_with_parcellation_annotation.csv"
    )
    pfc_parents = ["PL", "ILA", "ORBl", "ORBm", "ORBvl", "ACAd", "ACAv", "FRP"]
    pfc_pat = "|".join(pfc_parents)
    df = df.filter(
        pl.col("parcellation_structure").str.contains(pfc_pat)
    )
    return df


def load_neural_activity_pfc() -> pl.DataFrame:
    df = pl.read_csv(DATA_PATH / "pfcmap" / "units.csv")
    pfc_areas = ["PL", "ILA", "ACAd", "ACAv", "ORBm", "ORBl", "ORBvl", "FRP"]
    df = df.filter(pl.col("area").is_in(pfc_areas))
    df = df.filter(pl.col("rate_mean").is_not_null())
    return df


print("Loading MERFISH metadata...")
merfish = load_merfish_pfc()
print(f"Loaded {len(merfish)} MERFISH cells in PFC")

print("\nLoading neural activity data...")
neural = load_neural_activity_pfc()
print(f"Loaded {len(neural)} neurons in PFC")

merfish_substructures = merfish["parcellation_substructure"].unique().to_list()
print(f"\nMERFISH substructures found: {sorted(merfish_substructures)}")

merfish = merfish.with_columns(
    pl.col("parcellation_substructure")
    .map_elements(lambda s: SUBSTRUCTURE_TO_PARENT.get(s, None), return_dtype=pl.Utf8)
    .alias("parent_region")
)
merfish = merfish.filter(pl.col("parent_region").is_not_null())

parent_regions_merfish = merfish["parent_region"].unique().to_list()
print(f"MERFISH parent regions: {sorted(parent_regions_merfish)}")

entropy_per_parent = {}
cell_count_per_parent = {}
for parent in parent_regions_merfish:
    sub = merfish.filter(pl.col("parent_region") == parent)
    subclass_counts = Counter(sub["subclass"].to_list())
    ent = shannon_entropy(subclass_counts)
    entropy_per_parent[parent] = ent
    cell_count_per_parent[parent] = len(sub)
    print(f"  {parent}: entropy={ent:.3f}, n_cells={len(sub)}, n_subclasses={len(subclass_counts)}")

neural = neural.with_columns(
    pl.col("region")
    .map_elements(lambda s: SUBSTRUCTURE_TO_PARENT.get(s, None), return_dtype=pl.Utf8)
    .alias("parent_region")
)
neural = neural.filter(pl.col("parent_region").is_not_null())

parent_regions_neural = neural["parent_region"].unique().to_list()
print(f"\nNeural activity parent regions: {sorted(parent_regions_neural)}")

rate_per_parent = {}
neuron_count_per_parent = {}
for parent in parent_regions_neural:
    sub = neural.filter(pl.col("parent_region") == parent)
    rate_per_parent[parent] = float(sub["rate_mean"].mean())
    neuron_count_per_parent[parent] = len(sub)
    print(f"  {parent}: mean_rate={rate_per_parent[parent]:.3f}, n_neurons={len(sub)}")

common_parents = sorted(set(entropy_per_parent.keys()) & set(rate_per_parent.keys()))
print(f"\nCommon parent regions: {common_parents}")

if len(common_parents) < 4:
    print("Too few common regions for correlation analysis")
    results = {
        "hypothesis_id": "H07",
        "status": "INCONCLUSIVE",
        "statistic": None,
        "p_value": None,
        "effect_size": None,
        "n_regions": len(common_parents),
        "notes": "Too few common regions for correlation"
    }
else:
    entropies = np.array([entropy_per_parent[r] for r in common_parents])
    rates = np.array([rate_per_parent[r] for r in common_parents])

    rho, p_value = spearmanr(entropies, rates)
    print(f"\n=== Spearman Correlation ===")
    print(f"Regions: {common_parents}")
    print(f"Entropies: {entropies}")
    print(f"Mean rates: {rates}")
    print(f"Spearman rho = {rho:.4f}, p = {p_value:.4f}")

    n_perm = 10000
    null_rhos = []
    for _ in range(n_perm):
        perm_rates = np.random.permutation(rates)
        r_null, _ = spearmanr(entropies, perm_rates)
        null_rhos.append(r_null)
    null_rhos = np.array(null_rhos)
    p_perm = float(np.mean(null_rhos <= rho))
    print(f"Permutation test p (rho <= observed): {p_perm:.4f}")

    pearson_r = float(np.corrcoef(entropies, rates)[0, 1])
    print(f"Pearson r = {pearson_r:.4f}")

    status = "CONFIRMED" if (p_value < 0.05 and rho < 0) else "REFUTED"

    region_details = []
    for r in common_parents:
        region_details.append({
            "region": r,
            "entropy": round(entropy_per_parent[r], 4),
            "mean_rate": round(rate_per_parent[r], 4),
            "n_merfish_cells": cell_count_per_parent[r],
            "n_neurons": neuron_count_per_parent[r],
        })

    results = {
        "hypothesis_id": "H07",
        "status": status,
        "spearman_rho": float(rho),
        "spearman_p": float(p_value),
        "permutation_p": float(p_perm),
        "pearson_r": float(pearson_r),
        "n_regions": len(common_parents),
        "region_details": region_details,
        "notes": f"rho={rho:.3f}, p={p_value:.4f}; {status}"
    }

print("\n" + json.dumps(results, indent=2))

out_path = OUTPUT_DIR / "results.json"
out_path.write_text(json.dumps(results, indent=2))

if len(common_parents) >= 4:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    entropies = np.array([entropy_per_parent[r] for r in common_parents])
    rates = np.array([rate_per_parent[r] for r in common_parents])

    axes[0].scatter(entropies, rates, s=120, zorder=5, edgecolors="black", linewidths=0.8)
    for i, r in enumerate(common_parents):
        axes[0].annotate(
            r, (entropies[i], rates[i]),
            textcoords="offset points", xytext=(8, 5), fontsize=9
        )
    m, b = np.polyfit(entropies, rates, 1)
    x_line = np.linspace(entropies.min() - 0.05, entropies.max() + 0.05, 100)
    axes[0].plot(x_line, m * x_line + b, "--", color="gray", alpha=0.7)
    axes[0].set_xlabel("Subclass Shannon entropy (bits)")
    axes[0].set_ylabel("Mean firing rate (Hz)")
    axes[0].set_title(
        f"Transcriptomic Diversity vs Firing Rate\n"
        f"Spearman ρ = {rho:.3f}, p = {p_value:.3f}"
    )

    x_pos = np.arange(len(common_parents))
    width = 0.35
    ent_vals = [entropy_per_parent[r] for r in common_parents]
    rate_vals = [rate_per_parent[r] for r in common_parents]

    ax2 = axes[1]
    ax2.bar(x_pos - width / 2, ent_vals, width, label="Entropy (bits)", color="#55A868", edgecolor="black")
    ax2_twin = ax2.twinx()
    ax2_twin.bar(x_pos + width / 2, rate_vals, width, label="Mean rate (Hz)", color="#C44E52", edgecolor="black", alpha=0.8)

    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(common_parents, rotation=45, ha="right")
    ax2.set_ylabel("Shannon entropy (bits)", color="#55A868")
    ax2_twin.set_ylabel("Mean firing rate (Hz)", color="#C44E52")
    ax2.set_title("Entropy & Firing Rate by Parent Region")

    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "figures" / "entropy_vs_rate.png", dpi=150)
    plt.savefig(OUTPUT_DIR / "figures" / "entropy_vs_rate.svg")

print(f"\nResults saved to {out_path}")
print("Figures saved to figures/")

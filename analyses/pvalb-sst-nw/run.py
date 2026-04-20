"""
Hypothesis H18: PFC subregions with higher Pvalb-to-Sst interneuron ratios
(from MERFISH) have higher NW (fast-spiking) neuron fractions (from
neural_activity), because Pvalb interneurons are the primary fast-spiking
cell type in cortex.

Branch: pvalb-sst-nw
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

PFC_SUBREGIONS = [
    "ILA5", "ILA6a",
    "ORBl5", "ORBl6a",
    "ORBm5", "ORBm6a",
    "ORBvl5", "ORBvl6a",
    "PL5", "PL6a",
    "PL2/3", "ORBl2/3", "ORBm2/3", "ORBvl2/3",
]


def load_merfish_pfc() -> pl.DataFrame:
    df = pl.read_csv(
        DATA_PATH / "merfish" / "cell_metadata_with_parcellation_annotation.csv"
    )
    df = df.filter(pl.col("parcellation_substructure").is_in(PFC_SUBREGIONS))
    return df


def load_neural_activity_pfc() -> pl.DataFrame:
    df = pl.read_csv(DATA_PATH / "pfcmap" / "units.csv")
    df = df.filter(pl.col("region").is_in(PFC_SUBREGIONS))
    df = df.filter(pl.col("utype").is_in(["ww", "nw"]))
    df = df.filter(pl.col("rate_mean").is_not_null())
    return df


print("=" * 60)
print("H18: Pvalb/Sst Ratio → NW Fraction")
print("=" * 60)

print("\nLoading MERFISH data...")
merfish = load_merfish_pfc()
print(f"Loaded {len(merfish)} cells")

pv_sst_ratio = {}
for region in PFC_SUBREGIONS:
    sub = merfish.filter(pl.col("parcellation_substructure") == region)
    if len(sub) < 50:
        continue
    subclass_counts = Counter(sub["subclass"].to_list())
    n_pvalb = sum(v for k, v in subclass_counts.items() if "Pvalb" in k)
    n_sst = sum(v for k, v in subclass_counts.items() if "Sst" in k)
    if n_sst == 0:
        continue
    ratio = n_pvalb / n_sst
    pv_sst_ratio[region] = ratio
    print(f"  {region}: Pvalb={n_pvalb}, Sst={n_sst}, ratio={ratio:.3f}")

print("\nLoading neural activity data...")
neural = load_neural_activity_pfc()
print(f"Loaded {len(neural)} neurons")

nw_frac = {}
for region in PFC_SUBREGIONS:
    sub = neural.filter(pl.col("region") == region)
    if len(sub) < 10:
        continue
    n_nw = len(sub.filter(pl.col("utype") == "nw"))
    nw_frac[region] = n_nw / len(sub)
    print(f"  {region}: NW frac={nw_frac[region]:.3f}, n={len(sub)}")

common = sorted(set(pv_sst_ratio.keys()) & set(nw_frac.keys()))
print(f"\nCommon regions: {common}")

if len(common) < 4:
    results = {"hypothesis_id": "H18", "status": "INCONCLUSIVE", "n_regions": len(common)}
else:
    ratio_vals = np.array([pv_sst_ratio[r] for r in common])
    nw_vals = np.array([nw_frac[r] for r in common])

    rho, p_val = spearmanr(ratio_vals, nw_vals)
    pearson_r = float(np.corrcoef(ratio_vals, nw_vals)[0, 1])

    print(f"\n=== Results ===")
    print(f"  Spearman rho = {rho:.4f}, p = {p_val:.4f}")
    print(f"  Pearson r = {pearson_r:.4f}")

    n_perm = 10000
    null_rhos = []
    for _ in range(n_perm):
        perm = np.random.permutation(nw_vals)
        r_null, _ = spearmanr(ratio_vals, perm)
        null_rhos.append(r_null)
    null_rhos = np.array(null_rhos)
    p_perm = float(np.mean(null_rhos >= rho))
    print(f"  Permutation p: {p_perm:.4f}")

    status = "CONFIRMED" if (p_val < 0.05 and rho > 0) else "REFUTED"

    region_details = []
    for r in common:
        region_details.append({
            "region": r,
            "pvalb_sst_ratio": round(pv_sst_ratio[r], 4),
            "nw_fraction": round(nw_frac[r], 4),
        })

    results = {
        "hypothesis_id": "H18",
        "status": status,
        "spearman_rho": float(rho),
        "spearman_p": float(p_val),
        "pearson_r": float(pearson_r),
        "permutation_p": float(p_perm),
        "n_regions": len(common),
        "region_details": region_details,
        "notes": f"rho={rho:.3f}, p={p_val:.4f}; {status}"
    }

print("\n" + json.dumps(results, indent=2))

out_path = OUTPUT_DIR / "results.json"
out_path.write_text(json.dumps(results, indent=2))

if len(common) >= 4:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].scatter(ratio_vals, nw_vals, s=120, zorder=5, edgecolors="black", linewidths=0.8)
    for i, r in enumerate(common):
        axes[0].annotate(r, (ratio_vals[i], nw_vals[i]), textcoords="offset points", xytext=(6, 4), fontsize=7)
    m, b = np.polyfit(ratio_vals, nw_vals, 1)
    x_line = np.linspace(ratio_vals.min() * 0.9, ratio_vals.max() * 1.1, 100)
    axes[0].plot(x_line, m * x_line + b, "--", color="gray", alpha=0.7)
    axes[0].set_xlabel("Pvalb/Sst ratio")
    axes[0].set_ylabel("NW (fast-spiking) fraction")
    axes[0].set_title(f"Pvalb/Sst Ratio vs NW Fraction\nρ = {rho:.3f}, p = {p_val:.3f}")

    x_pos = np.arange(len(common))
    width = 0.35
    axes[1].bar(x_pos - width/2, ratio_vals, width, label="Pvalb/Sst", color="#8172B2", edgecolor="black")
    ax2 = axes[1].twinx()
    ax2.bar(x_pos + width/2, nw_vals, width, label="NW frac", color="#C44E52", edgecolor="black", alpha=0.8)
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels(common, rotation=45, ha="right", fontsize=7)
    axes[1].set_ylabel("Pvalb/Sst ratio", color="#8172B2")
    ax2.set_ylabel("NW fraction", color="#C44E52")
    axes[1].set_title("Pvalb/Sst & NW by Subregion")
    lines1, labels1 = axes[1].get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    axes[1].legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "figures" / "pvalb_sst_nw.png", dpi=150)
    plt.savefig(OUTPUT_DIR / "figures" / "pvalb_sst_nw.svg")

print(f"\nResults saved to {out_path}")

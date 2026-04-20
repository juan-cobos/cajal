"""
Hypothesis H16: The burst index difference between NW and WW neurons is
larger in Layer 5 than Layer 6 PFC subregions — L5 interneurons (higher GABA
fraction from H15) show more pronounced burst-like firing differences from
regular-spiking neurons.

Branch: burst-layer-utype
Datasets: neural_activity
"""

import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from scipy.stats import mannwhitneyu

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

DATA_PATH = Path(__file__).parent.parent.parent / "data"
OUTPUT_DIR = Path(__file__).parent

L5_REGIONS = ["PL5", "ILA5", "ORBl5", "ORBm5", "ORBvl5"]
L6_REGIONS = ["PL6a", "ILA6a", "ORBl6a", "ORBm6a", "ORBvl6a"]
PARENT_NAMES = ["PL", "ILA", "ORBl", "ORBm", "ORBvl"]


def load_neural_activity_pfc() -> pl.DataFrame:
    df = pl.read_csv(DATA_PATH / "pfcmap" / "units.csv")
    all_regions = L5_REGIONS + L6_REGIONS
    df = df.filter(pl.col("region").is_in(all_regions))
    df = df.filter(pl.col("utype").is_in(["ww", "nw"]))
    df = df.filter(pl.col("B_mean").is_not_null())
    df = df.filter(pl.col("rate_mean").is_not_null())
    return df


print("=" * 60)
print("H16: NW-WW Burst Difference: L5 vs L6")
print("=" * 60)

print("\nLoading neural activity data...")
df = load_neural_activity_pfc()
print(f"Loaded {len(df)} neurons")

l5_diffs = []
l6_diffs = []
region_results = []

for l5, l6, parent in zip(L5_REGIONS, L6_REGIONS, PARENT_NAMES):
    for layer_label, region in [("L5", l5), ("L6", l6)]:
        sub = df.filter(pl.col("region") == region)
        nw = sub.filter(pl.col("utype") == "nw")
        ww = sub.filter(pl.col("utype") == "ww")
        if len(nw) < 5 or len(ww) < 5:
            continue
        nw_burst = float(nw["B_mean"].mean())
        ww_burst = float(ww["B_mean"].mean())
        nw_nw_frac = len(nw) / len(sub)
        diff = nw_burst - ww_burst
        region_results.append({
            "region": region,
            "layer": layer_label,
            "parent": parent,
            "nw_burst": nw_burst,
            "ww_burst": ww_burst,
            "diff": diff,
            "n_nw": len(nw),
            "n_ww": len(ww),
            "nw_frac": nw_nw_frac,
        })
        print(f"  {region} ({layer_label}): NW burst={nw_burst:.4f}, WW burst={ww_burst:.4f}, diff={diff:+.4f}")

    l5_sub = df.filter(pl.col("region") == l5)
    l6_sub = df.filter(pl.col("region") == l6)
    l5_nw = l5_sub.filter(pl.col("utype") == "nw")
    l5_ww = l5_sub.filter(pl.col("utype") == "ww")
    l6_nw = l6_sub.filter(pl.col("utype") == "nw")
    l6_ww = l6_sub.filter(pl.col("utype") == "ww")

    if len(l5_nw) >= 5 and len(l5_ww) >= 5 and len(l6_nw) >= 5 and len(l6_ww) >= 5:
        l5_diff = float(l5_nw["B_mean"].mean()) - float(l5_ww["B_mean"].mean())
        l6_diff = float(l6_nw["B_mean"].mean()) - float(l6_ww["B_mean"].mean())
        l5_diffs.append(l5_diff)
        l6_diffs.append(l6_diff)
        print(f"  → {parent}: L5 diff={l5_diff:+.4f}, L6 diff={l6_diff:+.4f}, L5>L6: {l5_diff > l6_diff}")

l5_diffs = np.array(l5_diffs)
l6_diffs = np.array(l6_diffs)
n_pairs = len(l5_diffs)

if n_pairs < 3:
    results = {"hypothesis_id": "H16", "status": "INCONCLUSIVE", "n_pairs": n_pairs}
else:
    from scipy.stats import wilcoxon
    stat, p_val = wilcoxon(l5_diffs, l6_diffs, alternative="greater")
    print(f"\n=== Wilcoxon signed-rank (L5 diff > L6 diff) ===")
    print(f"  stat={stat:.1f}, p={p_val:.4f}")
    print(f"  Mean L5 diff: {l5_diffs.mean():.4f}")
    print(f"  Mean L6 diff: {l6_diffs.mean():.4f}")
    print(f"  Mean diff-of-diffs: {(l5_diffs - l6_diffs).mean():.4f}")

    n_perm = 10000
    null_stats = []
    pair_diffs = l5_diffs - l6_diffs
    for _ in range(n_perm):
        signs = np.random.choice([-1, 1], size=n_pairs)
        null_stats.append(np.mean(signs * pair_diffs))
    null_stats = np.array(null_stats)
    p_perm = float(np.mean(null_stats >= np.mean(pair_diffs)))
    print(f"  Permutation p: {p_perm:.4f}")

    n_l5_greater = int(np.sum(l5_diffs > l6_diffs))
    print(f"  Pairs where L5 diff > L6 diff: {n_l5_greater}/{n_pairs}")

    status = "CONFIRMED" if p_val < 0.05 else "REFUTED"

    results = {
        "hypothesis_id": "H16",
        "status": status,
        "wilcoxon_p": float(p_val),
        "permutation_p": float(p_perm),
        "mean_l5_diff": float(l5_diffs.mean()),
        "mean_l6_diff": float(l6_diffs.mean()),
        "n_pairs": n_pairs,
        "n_l5_greater": n_l5_greater,
        "region_details": region_results,
        "notes": f"L5 diff={l5_diffs.mean():.4f} vs L6 diff={l6_diffs.mean():.4f}, p={p_val:.4f}; {status}"
    }

print("\n" + json.dumps(results, indent=2))

out_path = OUTPUT_DIR / "results.json"
out_path.write_text(json.dumps(results, indent=2))

if n_pairs >= 3:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    x_pos = np.arange(n_pairs)
    width = 0.35
    axes[0].bar(x_pos - width/2, l5_diffs, width, label="Layer 5 (NW-WW)", color="#DD8452", edgecolor="black")
    axes[0].bar(x_pos + width/2, l6_diffs, width, label="Layer 6 (NW-WW)", color="#4C72B0", edgecolor="black")
    axes[0].set_xticks(x_pos)
    axes[0].set_xticklabels(PARENT_NAMES[:n_pairs])
    axes[0].set_ylabel("Burst index difference (NW - WW)")
    axes[0].set_title("NW-WW Burst Difference by Layer")
    axes[0].legend()
    axes[0].axhline(0, color="black", linestyle="--", alpha=0.5)

    l5_data = [r for r in region_results if r["layer"] == "L5"]
    l6_data = [r for r in region_results if r["layer"] == "L6"]
    axes[1].scatter(
        [r["nw_frac"] for r in l5_data], [r["diff"] for r in l5_data],
        s=80, color="#DD8452", label="L5", edgecolors="black"
    )
    axes[1].scatter(
        [r["nw_frac"] for r in l6_data], [r["diff"] for r in l6_data],
        s=80, color="#4C72B0", label="L6", edgecolors="black"
    )
    for r in region_results:
        axes[1].annotate(r["region"], (r["nw_frac"], r["diff"]), textcoords="offset points", xytext=(4, 3), fontsize=7)
    axes[1].set_xlabel("NW fraction")
    axes[1].set_ylabel("NW-WW burst difference")
    axes[1].set_title("NW Fraction vs Burst Difference")
    axes[1].legend()
    axes[1].axhline(0, color="black", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "figures" / "burst_layer_utype.png", dpi=150)
    plt.savefig(OUTPUT_DIR / "figures" / "burst_layer_utype.svg")

print(f"\nResults saved to {out_path}")

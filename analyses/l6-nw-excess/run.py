"""
Hypothesis H19: Layer 6 PFC subregions have higher NW (fast-spiking) neuron
fractions than Layer 5 counterparts, because L6 corticothalamic projection
neurons require fast temporal precision for thalamic feedback signaling.

This directly tests the layer-NW relationship observed in H13/H18 as a
paradox, now formalized as a hypothesis.

Branch: l6-nw-excess
Datasets: neural_activity
"""

import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from scipy.stats import wilcoxon

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
    return df


print("=" * 60)
print("H19: L6 > L5 NW Fraction in PFC")
print("=" * 60)

print("\nLoading neural activity data...")
df = load_neural_activity_pfc()
print(f"Loaded {len(df)} neurons")

l5_nw = []
l6_nw = []
pairs = []

for l5, l6, parent in zip(L5_REGIONS, L6_REGIONS, PARENT_NAMES):
    l5_sub = df.filter(pl.col("region") == l5)
    l6_sub = df.filter(pl.col("region") == l6)
    if len(l5_sub) < 10 or len(l6_sub) < 10:
        continue
    l5_frac = len(l5_sub.filter(pl.col("utype") == "nw")) / len(l5_sub)
    l6_frac = len(l6_sub.filter(pl.col("utype") == "nw")) / len(l6_sub)
    l5_nw.append(l5_frac)
    l6_nw.append(l6_frac)
    pairs.append({"parent": parent, "l5_nw": l5_frac, "l6_nw": l6_frac, "diff": l6_frac - l5_frac})
    print(f"  {parent}: L5 NW={l5_frac:.3f}, L6 NW={l6_frac:.3f}, diff={l6_frac-l5_frac:+.3f}")

l5_nw = np.array(l5_nw)
l6_nw = np.array(l6_nw)

stat, p_val = wilcoxon(l6_nw, l5_nw, alternative="greater")
print(f"\nWilcoxon signed-rank (L6 > L5 NW): stat={stat:.1f}, p={p_val:.4f}")

n_perm = 10000
null_stats = []
diffs = l6_nw - l5_nw
for _ in range(n_perm):
    signs = np.random.choice([-1, 1], size=len(diffs))
    null_stats.append(np.mean(signs * diffs))
null_stats = np.array(null_stats)
p_perm = float(np.mean(null_stats >= diffs.mean()))
print(f"Permutation p: {p_perm:.4f}")

n_l6_greater = int(np.sum(l6_nw > l5_nw))
print(f"Pairs where L6 > L5: {n_l6_greater}/{len(pairs)}")

status = "CONFIRMED" if p_val < 0.05 else "REFUTED"

results = {
    "hypothesis_id": "H19",
    "status": status,
    "wilcoxon_p": float(p_val),
    "permutation_p": float(p_perm),
    "mean_l5_nw": float(l5_nw.mean()),
    "mean_l6_nw": float(l6_nw.mean()),
    "mean_diff": float((l6_nw - l5_nw).mean()),
    "n_pairs": len(pairs),
    "n_l6_greater": n_l6_greater,
    "pairs": pairs,
    "notes": f"L6 NW={l6_nw.mean():.3f} > L5 NW={l5_nw.mean():.3f}, p={p_val:.4f}; {status}"
}

print("\n" + json.dumps(results, indent=2))

out_path = OUTPUT_DIR / "results.json"
out_path.write_text(json.dumps(results, indent=2))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

x_pos = np.arange(len(pairs))
width = 0.35
axes[0].bar(x_pos - width/2, l5_nw, width, label="Layer 5", color="#DD8452", edgecolor="black")
axes[0].bar(x_pos + width/2, l6_nw, width, label="Layer 6", color="#4C72B0", edgecolor="black")
axes[0].set_xticks(x_pos)
axes[0].set_xticklabels(PARENT_NAMES)
axes[0].set_ylabel("NW (fast-spiking) fraction")
axes[0].set_title(f"NW Fraction: L6 vs L5\nWilcoxon p = {p_val:.4f}")
axes[0].legend()

diffs_plot = l6_nw - l5_nw
axes[1].bar(x_pos, diffs_plot, color=["#4C72B0" if d > 0 else "#DD8452" for d in diffs_plot], edgecolor="black")
axes[1].set_xticks(x_pos)
axes[1].set_xticklabels(PARENT_NAMES)
axes[1].set_ylabel("L6 - L5 NW fraction")
axes[1].set_title("L6 - L5 NW Fraction Difference")
axes[1].axhline(0, color="black", linestyle="--", alpha=0.5)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "figures" / "l6_nw_excess.png", dpi=150)
plt.savefig(OUTPUT_DIR / "figures" / "l6_nw_excess.svg")

print(f"\nResults saved to {out_path}")

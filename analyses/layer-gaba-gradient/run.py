"""
Hypothesis H17: GABAergic cell fraction follows a laminar gradient
L2/3 < L5 > L6 (inverted-U) across PFC subregions, with Layer 5 as the
peak interneuron density layer — consistent with known cortical microcircuit
architecture where L5 receives the densest thalamic and intracortical input.

Branch: layer-gaba-gradient
Datasets: merfish
"""

import json
import random
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from scipy.stats import friedmanchisquare

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

DATA_PATH = Path(__file__).parent.parent.parent / "data"
OUTPUT_DIR = Path(__file__).parent

LAYER_REGIONS = {
    "L2/3": ["PL2/3", "ORBl2/3", "ORBm2/3", "ORBvl2/3"],
    "L5": ["PL5", "ILA5", "ORBl5", "ORBm5", "ORBvl5"],
    "L6": ["PL6a", "ILA6a", "ORBl6a", "ORBm6a", "ORBvl6a"],
}

PARENT_MAP = {
    "PL2/3": "PL", "PL5": "PL", "PL6a": "PL",
    "ORBl2/3": "ORBl", "ORBl5": "ORBl", "ORBl6a": "ORBl",
    "ORBm2/3": "ORBm", "ORBm5": "ORBm", "ORBm6a": "ORBm",
    "ORBvl2/3": "ORBvl", "ORBvl5": "ORBvl", "ORBvl6a": "ORBvl",
}

COMMON_PARENTS = ["PL", "ORBl", "ORBm", "ORBvl"]


def load_merfish_pfc() -> pl.DataFrame:
    df = pl.read_csv(
        DATA_PATH / "merfish" / "cell_metadata_with_parcellation_annotation.csv"
    )
    all_regions = []
    for regions in LAYER_REGIONS.values():
        all_regions.extend(regions)
    df = df.filter(pl.col("parcellation_substructure").is_in(all_regions))
    return df


print("=" * 60)
print("H17: Laminar GABA Gradient L2/3 < L5 > L6")
print("=" * 60)

print("\nLoading MERFISH data...")
df = load_merfish_pfc()
print(f"Loaded {len(df)} cells")

gaba_frac = {}
for layer, regions in LAYER_REGIONS.items():
    for region in regions:
        sub = df.filter(pl.col("parcellation_substructure") == region)
        if len(sub) == 0:
            continue
        nt_counts = Counter(sub["neurotransmitter"].to_list())
        n_gaba = nt_counts.get("GABA", 0)
        n_glut = nt_counts.get("Glut", 0)
        n_total = n_gaba + n_glut
        if n_total == 0:
            continue
        gaba_frac[region] = n_gaba / n_total
        print(f"  {region} ({layer}): GABA={gaba_frac[region]:.3f}, n={len(sub)}")

l23_vals = []
l5_vals = []
l6_vals = []
parent_labels = []

for parent in COMMON_PARENTS:
    l23_r = f"{parent}2/3"
    l5_r = f"{parent}5"
    l6_r = f"{parent}6a"
    if l23_r in gaba_frac and l5_r in gaba_frac and l6_r in gaba_frac:
        l23_vals.append(gaba_frac[l23_r])
        l5_vals.append(gaba_frac[l5_r])
        l6_vals.append(gaba_frac[l6_r])
        parent_labels.append(parent)
        print(f"  {parent}: L2/3={gaba_frac[l23_r]:.3f}, L5={gaba_frac[l5_r]:.3f}, L6={gaba_frac[l6_r]:.3f}")

l23_vals = np.array(l23_vals)
l5_vals = np.array(l5_vals)
l6_vals = np.array(l6_vals)
n_parents = len(parent_labels)

print(f"\n=== Paired Analysis ({n_parents} parents) ===")

stat_friedman, p_friedman = friedmanchisquare(l23_vals, l5_vals, l6_vals)
print(f"Friedman test: chi2={stat_friedman:.3f}, p={p_friedman:.4f}")

from scipy.stats import wilcoxon
stat_l5_l23, p_l5_l23 = wilcoxon(l5_vals, l23_vals, alternative="greater")
stat_l5_l6, p_l5_l6 = wilcoxon(l5_vals, l6_vals, alternative="greater")
print(f"L5 > L2/3: Wilcoxon p={p_l5_l23:.4f}")
print(f"L5 > L6: Wilcoxon p={p_l5_l6:.4f}")

inverted_u = all(l5_vals[i] > l23_vals[i] and l5_vals[i] > l6_vals[i] for i in range(n_parents))
n_inverted_u = sum(1 for i in range(n_parents) if l5_vals[i] > l23_vals[i] and l5_vals[i] > l6_vals[i])
print(f"Regions with inverted-U (L2/3 < L5 > L6): {n_inverted_u}/{n_parents}")

n_perm = 10000
null_u = []
for _ in range(n_perm):
    signs1 = np.random.choice([-1, 1], size=n_parents)
    signs2 = np.random.choice([-1, 1], size=n_parents)
    perm_l5_l23 = signs1 * (l5_vals - l23_vals)
    perm_l5_l6 = signs2 * (l5_vals - l6_vals)
    n_u_perm = sum(1 for i in range(n_parents) if perm_l5_l23[i] > 0 and perm_l5_l6[i] > 0)
    null_u.append(n_u_perm)
null_u = np.array(null_u)
p_perm = float(np.mean(null_u >= n_inverted_u))
print(f"Permutation p (>= {n_inverted_u} inverted-U): {p_perm:.4f}")

status = "CONFIRMED" if (p_friedman < 0.05 and n_inverted_u == n_parents) else "REFUTED"

results = {
    "hypothesis_id": "H17",
    "status": status,
    "friedman_p": float(p_friedman),
    "l5_gt_l23_p": float(p_l5_l23),
    "l5_gt_l6_p": float(p_l5_l6),
    "n_inverted_u": n_inverted_u,
    "n_parents": n_parents,
    "permutation_p": float(p_perm),
    "mean_l23_gaba": float(l23_vals.mean()),
    "mean_l5_gaba": float(l5_vals.mean()),
    "mean_l6_gaba": float(l6_vals.mean()),
    "layer_details": [
        {"parent": p, "l23": round(l23_vals[i], 4), "l5": round(l5_vals[i], 4), "l6": round(l6_vals[i], 4)}
        for i, p in enumerate(parent_labels)
    ],
    "notes": f"L2/3={l23_vals.mean():.3f} < L5={l5_vals.mean():.3f} > L6={l6_vals.mean():.3f}; Friedman p={p_friedman:.4f}; {n_inverted_u}/{n_parents} inverted-U; {status}"
}

print("\n" + json.dumps(results, indent=2))

out_path = OUTPUT_DIR / "results.json"
out_path.write_text(json.dumps(results, indent=2))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

x_pos = np.arange(n_parents)
width = 0.25
axes[0].bar(x_pos - width, l23_vals, width, label="L2/3", color="#55A868", edgecolor="black")
axes[0].bar(x_pos, l5_vals, width, label="L5", color="#DD8452", edgecolor="black")
axes[0].bar(x_pos + width, l6_vals, width, label="L6", color="#4C72B0", edgecolor="black")
axes[0].set_xticks(x_pos)
axes[0].set_xticklabels(parent_labels)
axes[0].set_ylabel("GABAergic fraction")
axes[0].set_title("GABA Fraction by Layer and Parent Region")
axes[0].legend()

bp = axes[1].boxplot([l23_vals, l5_vals, l6_vals], tick_labels=["L2/3", "L5", "L6"], patch_artist=True)
for i, box in enumerate(bp["boxes"]):
    box.set_facecolor(["#55A868", "#DD8452", "#4C72B0"][i])
    box.set_alpha(0.7)
axes[1].set_ylabel("GABAergic fraction")
axes[1].set_title(f"Laminar GABA Gradient\nFriedman p = {p_friedman:.4f}")

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "figures" / "layer_gaba_gradient.png", dpi=150)
plt.savefig(OUTPUT_DIR / "figures" / "layer_gaba_gradient.svg")

print(f"\nResults saved to {out_path}")

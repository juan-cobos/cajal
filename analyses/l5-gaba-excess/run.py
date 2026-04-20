"""
Hypothesis H15: Layer 5 PFC subregions have significantly higher GABAergic
cell fractions than their Layer 6 counterparts within the same parent region,
reflecting the higher inhibitory interneuron density in the input-rich Layer 5
cortical microcircuit.

Branch: l5-gaba-excess
Datasets: merfish
"""

import json
import random
from collections import Counter
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

L2_3_REGIONS = ["PL2/3", "ORBl2/3", "ORBm2/3", "ORBvl2/3"]


def load_merfish_pfc() -> pl.DataFrame:
    df = pl.read_csv(
        DATA_PATH / "merfish" / "cell_metadata_with_parcellation_annotation.csv"
    )
    all_regions = L5_REGIONS + L6_REGIONS + L2_3_REGIONS
    df = df.filter(pl.col("parcellation_substructure").is_in(all_regions))
    return df


print("=" * 60)
print("H15: Layer 5 > Layer 6 GABAergic Fraction in PFC")
print("=" * 60)

print("\nLoading MERFISH data...")
df = load_merfish_pfc()
print(f"Loaded {len(df)} cells")

gaba_frac = {}
for region in L5_REGIONS + L6_REGIONS + L2_3_REGIONS:
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

    subclass_counts = Counter(sub["subclass"].to_list())
    n_pvalb = sum(v for k, v in subclass_counts.items() if "Pvalb" in k)
    n_sst = sum(v for k, v in subclass_counts.items() if "Sst" in k)
    n_vip = sum(v for k, v in subclass_counts.items() if "Vip" in k)
    print(f"  {region}: GABA={gaba_frac[region]:.3f}, Pvalb={n_pvalb}, Sst={n_sst}, Vip={n_vip}")

l5_gaba = np.array([gaba_frac[r] for r in L5_REGIONS if r in gaba_frac])
l6_gaba = np.array([gaba_frac[r] for r in L6_REGIONS if r in gaba_frac])

print(f"\n=== Primary Test: L5 vs L6 GABA fraction (paired by parent) ===")
pairs = []
for l5, l6, parent in zip(L5_REGIONS, L6_REGIONS, PARENT_NAMES):
    if l5 in gaba_frac and l6 in gaba_frac:
        pairs.append((gaba_frac[l5], gaba_frac[l6], parent))
        print(f"  {parent}: L5={gaba_frac[l5]:.3f}, L6={gaba_frac[l6]:.3f}, diff={gaba_frac[l5]-gaba_frac[l6]:+.3f}")

l5_vals = np.array([p[0] for p in pairs])
l6_vals = np.array([p[1] for p in pairs])
diffs = l5_vals - l6_vals

stat, p_val = wilcoxon(l5_vals, l6_vals, alternative="greater")
print(f"\nWilcoxon signed-rank (L5 > L6 GABA): stat={stat:.1f}, p={p_val:.4f}")
print(f"Mean diff: {diffs.mean():.4f}")

n_perm = 10000
null_diffs = []
for _ in range(n_perm):
    signs = np.random.choice([-1, 1], size=len(pairs))
    perm_diffs = signs * diffs
    null_diffs.append(perm_diffs.mean())
null_diffs = np.array(null_diffs)
p_perm = float(np.mean(null_diffs >= diffs.mean()))
print(f"Permutation p: {p_perm:.4f}")

status = "CONFIRMED" if p_val < 0.05 else "REFUTED"

l23_gaba_vals = [gaba_frac[r] for r in L2_3_REGIONS if r in gaba_frac]
l23_mean = np.mean(l23_gaba_vals) if l23_gaba_vals else None
print(f"\nLayer 2/3 mean GABA fraction: {l23_mean:.3f}" if l23_mean else "")

results = {
    "hypothesis_id": "H15",
    "status": status,
    "wilcoxon_p": float(p_val),
    "permutation_p": float(p_perm),
    "mean_l5_gaba": float(l5_vals.mean()),
    "mean_l6_gaba": float(l6_vals.mean()),
    "mean_diff": float(diffs.mean()),
    "n_pairs": len(pairs),
    "pairs": [{"parent": p[2], "l5_gaba": round(p[0], 4), "l6_gaba": round(p[1], 4), "diff": round(p[0]-p[1], 4)} for p in pairs],
    "l23_mean_gaba": float(l23_mean) if l23_mean else None,
    "notes": f"L5 GABA={l5_vals.mean():.3f} > L6={l6_vals.mean():.3f}, p={p_val:.4f}; {status}"
}

print("\n" + json.dumps(results, indent=2))

out_path = OUTPUT_DIR / "results.json"
out_path.write_text(json.dumps(results, indent=2))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

x_pos = np.arange(len(pairs))
width = 0.35
axes[0].bar(x_pos - width/2, l5_vals, width, label="Layer 5", color="#DD8452", edgecolor="black")
axes[0].bar(x_pos + width/2, l6_vals, width, label="Layer 6", color="#4C72B0", edgecolor="black")
axes[0].set_xticks(x_pos)
axes[0].set_xticklabels([p[2] for p in pairs])
axes[0].set_ylabel("GABAergic fraction")
axes[0].set_title(f"GABA Fraction: L5 vs L6\nWilcoxon p = {p_val:.4f}")
axes[0].legend()

all_layers_gaba = []
all_layers_labels = []
for r in L2_3_REGIONS:
    if r in gaba_frac:
        all_layers_gaba.append(gaba_frac[r])
        all_layers_labels.append("L2/3")
for r in L5_REGIONS:
    if r in gaba_frac:
        all_layers_gaba.append(gaba_frac[r])
        all_layers_labels.append("L5")
for r in L6_REGIONS:
    if r in gaba_frac:
        all_layers_gaba.append(gaba_frac[r])
        all_layers_labels.append("L6")

layer_data = {"L2/3": [], "L5": [], "L6": []}
for g, l in zip(all_layers_gaba, all_layers_labels):
    layer_data[l].append(g)

bp = axes[1].boxplot(
    [layer_data["L2/3"], layer_data["L5"], layer_data["L6"]],
    tick_labels=["Layer 2/3", "Layer 5", "Layer 6"],
    patch_artist=True,
)
colors = ["#55A868", "#DD8452", "#4C72B0"]
for i, box in enumerate(bp["boxes"]):
    box.set_facecolor(colors[i])
    box.set_alpha(0.7)
axes[1].set_ylabel("GABAergic fraction")
axes[1].set_title("GABA Fraction by Cortical Layer")

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "figures" / "l5_gaba_excess.png", dpi=150)
plt.savefig(OUTPUT_DIR / "figures" / "l5_gaba_excess.svg")

print(f"\nResults saved to {out_path}")

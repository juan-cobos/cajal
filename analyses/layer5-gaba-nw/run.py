"""
Hypothesis H13: Within PFC subregions, Layer 5 has a higher GABAergic cell
fraction than Layer 6 (from MERFISH), and this L5>L6 GABA excess correlates
with the L5>L6 NW-fraction difference across subregions (from neural_activity).

This uses a within-region paired contrast (L5 vs L6) across 5 parent regions,
yielding 5 paired differences — testing whether local inhibitory composition
tracks electrophysiological subtype across layers.

Branch: layer5-gaba-nw
Datasets: merfish, neural_activity
"""

import json
import random
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from scipy.stats import spearmanr, wilcoxon

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

DATA_PATH = Path(__file__).parent.parent.parent / "data"
OUTPUT_DIR = Path(__file__).parent

L5_REGIONS = ["PL5", "ILA5", "ORBl5", "ORBm5", "ORBvl5"]
L6_REGIONS = ["PL6a", "ILA6a", "ORBl6a", "ORBm6a", "ORBvl6a"]
LAYER_MAP = {r5: r6 for r5, r6 in zip(L5_REGIONS, L6_REGIONS)}

PARENT_NAMES = ["PL", "ILA", "ORBl", "ORBm", "ORBvl"]


def load_merfish_pfc() -> pl.DataFrame:
    df = pl.read_csv(
        DATA_PATH / "merfish" / "cell_metadata_with_parcellation_annotation.csv"
    )
    all_regions = L5_REGIONS + L6_REGIONS
    df = df.filter(pl.col("parcellation_substructure").is_in(all_regions))
    return df


def load_neural_activity_pfc() -> pl.DataFrame:
    df = pl.read_csv(DATA_PATH / "pfcmap" / "units.csv")
    all_regions = L5_REGIONS + L6_REGIONS
    df = df.filter(pl.col("region").is_in(all_regions))
    df = df.filter(pl.col("utype").is_in(["ww", "nw"]))
    df = df.filter(pl.col("rate_mean").is_not_null())
    return df


print("=" * 60)
print("H13: Layer 5 vs Layer 6 GABA Fraction & NW Fraction")
print("=" * 60)

print("\n[1] Loading MERFISH data...")
merfish = load_merfish_pfc()
print(f"  Loaded {len(merfish)} cells")

gaba_frac = {}
for region in L5_REGIONS + L6_REGIONS:
    sub = merfish.filter(pl.col("parcellation_substructure") == region)
    if len(sub) == 0:
        continue
    nt_counts = Counter(sub["neurotransmitter"].to_list())
    n_gaba = nt_counts.get("GABA", 0)
    n_glut = nt_counts.get("Glut", 0)
    n_total = n_gaba + n_glut
    if n_total == 0:
        continue
    gaba_frac[region] = n_gaba / n_total
    layer = "L5" if region in L5_REGIONS else "L6"
    print(f"  {region} ({layer}): GABA frac = {gaba_frac[region]:.3f}")

print("\n[2] Loading neural activity data...")
neural = load_neural_activity_pfc()
print(f"  Loaded {len(neural)} neurons")

nw_frac = {}
for region in L5_REGIONS + L6_REGIONS:
    sub = neural.filter(pl.col("region") == region)
    if len(sub) == 0:
        continue
    n_nw = len(sub.filter(pl.col("utype") == "nw"))
    n_total = len(sub)
    nw_frac[region] = n_nw / n_total
    layer = "L5" if region in L5_REGIONS else "L6"
    print(f"  {region} ({layer}): NW frac = {nw_frac[region]:.3f}, n = {n_total}")

print("\n[3] Computing L5 vs L6 paired differences...")
gaba_diff = []
nw_diff = []
parent_labels = []

for l5, l6 in zip(L5_REGIONS, L6_REGIONS):
    if l5 not in gaba_frac or l6 not in gaba_frac:
        continue
    if l5 not in nw_frac or l6 not in nw_frac:
        continue
    g_diff = gaba_frac[l5] - gaba_frac[l6]
    n_diff = nw_frac[l5] - nw_frac[l6]
    parent = PARENT_NAMES[L5_REGIONS.index(l5)]
    gaba_diff.append(g_diff)
    nw_diff.append(n_diff)
    parent_labels.append(parent)
    print(f"  {parent}: GABA L5-L6 = {g_diff:+.3f}, NW L5-L6 = {n_diff:+.3f}")

gaba_diff = np.array(gaba_diff)
nw_diff = np.array(nw_diff)
n_pairs = len(gaba_diff)

if n_pairs < 3:
    results = {"hypothesis_id": "H13", "status": "INCONCLUSIVE", "n_pairs": n_pairs}
else:
    rho, p_val = spearmanr(gaba_diff, nw_diff)
    pearson_r = float(np.corrcoef(gaba_diff, nw_diff)[0, 1])
    print(f"\n=== Correlation ===")
    print(f"  Spearman rho = {rho:.4f}, p = {p_val:.4f}")
    print(f"  Pearson r = {pearson_r:.4f}")

    stat_l5_gt_l6_gaba, p_l5_gt_l6_gaba = wilcoxon(gaba_diff, alternative="greater")
    stat_l5_gt_l6_nw, p_l5_gt_l6_nw = wilcoxon(nw_diff, alternative="greater")
    print(f"  L5 > L6 GABA fraction: Wilcoxon p = {p_l5_gt_l6_gaba:.4f}")
    print(f"  L5 > L6 NW fraction: Wilcoxon p = {p_l5_gt_l6_nw:.4f}")

    status = "CONFIRMED" if (p_val < 0.05 and rho > 0) else "REFUTED"

    region_details = []
    for i, parent in enumerate(parent_labels):
        l5 = L5_REGIONS[PARENT_NAMES.index(parent)]
        l6 = L6_REGIONS[PARENT_NAMES.index(parent)]
        region_details.append({
            "parent": parent,
            "l5_gaba": round(gaba_frac[l5], 4),
            "l6_gaba": round(gaba_frac[l6], 4),
            "gaba_diff": round(gaba_diff[i], 4),
            "l5_nw": round(nw_frac[l5], 4),
            "l6_nw": round(nw_frac[l6], 4),
            "nw_diff": round(nw_diff[i], 4),
        })

    results = {
        "hypothesis_id": "H13",
        "status": status,
        "spearman_rho": float(rho),
        "spearman_p": float(p_val),
        "pearson_r": float(pearson_r),
        "n_pairs": n_pairs,
        "l5_gt_l6_gaba_p": float(p_l5_gt_l6_gaba),
        "l5_gt_l6_nw_p": float(p_l5_gt_l6_nw),
        "mean_gaba_diff": float(gaba_diff.mean()),
        "mean_nw_diff": float(nw_diff.mean()),
        "region_details": region_details,
        "notes": f"rho={rho:.3f} p={p_val:.3f}; GABA diff mean={gaba_diff.mean():.3f}; NW diff mean={nw_diff.mean():.3f}; {status}"
    }

print("\n" + json.dumps(results, indent=2))

out_path = OUTPUT_DIR / "results.json"
out_path.write_text(json.dumps(results, indent=2))

if n_pairs >= 3:
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))

    axes[0].scatter(gaba_diff, nw_diff, s=150, zorder=5, edgecolors="black", linewidths=0.8)
    for i, parent in enumerate(parent_labels):
        axes[0].annotate(parent, (gaba_diff[i], nw_diff[i]), textcoords="offset points", xytext=(8, 5), fontsize=10)
    m, b = np.polyfit(gaba_diff, nw_diff, 1)
    x_line = np.linspace(gaba_diff.min() - 0.01, gaba_diff.max() + 0.01, 100)
    axes[0].plot(x_line, m * x_line + b, "--", color="gray", alpha=0.7)
    axes[0].axhline(0, color="black", linestyle=":", alpha=0.3)
    axes[0].axvline(0, color="black", linestyle=":", alpha=0.3)
    axes[0].set_xlabel("GABA fraction diff (L5 - L6)")
    axes[0].set_ylabel("NW fraction diff (L5 - L6)")
    axes[0].set_title(f"L5-L6: GABA vs NW Difference\nρ = {rho:.3f}, p = {p_val:.3f}")

    x_pos = np.arange(n_pairs)
    width = 0.3
    l5_gaba = [gaba_frac[L5_REGIONS[i]] for i in range(n_pairs) if L5_REGIONS[i] in gaba_frac]
    l6_gaba = [gaba_frac[L6_REGIONS[i]] for i in range(n_pairs) if L6_REGIONS[i] in gaba_frac]
    axes[1].bar(x_pos - width/2, l5_gaba, width, label="Layer 5", color="#DD8452", edgecolor="black")
    axes[1].bar(x_pos + width/2, l6_gaba, width, label="Layer 6", color="#4C72B0", edgecolor="black")
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels(parent_labels)
    axes[1].set_ylabel("GABA fraction")
    axes[1].set_title("GABA Fraction by Layer")
    axes[1].legend()

    l5_nw = [nw_frac[L5_REGIONS[i]] for i in range(n_pairs) if L5_REGIONS[i] in nw_frac]
    l6_nw = [nw_frac[L6_REGIONS[i]] for i in range(n_pairs) if L6_REGIONS[i] in nw_frac]
    axes[2].bar(x_pos - width/2, l5_nw, width, label="Layer 5", color="#DD8452", edgecolor="black")
    axes[2].bar(x_pos + width/2, l6_nw, width, label="Layer 6", color="#4C72B0", edgecolor="black")
    axes[2].set_xticks(x_pos)
    axes[2].set_xticklabels(parent_labels)
    axes[2].set_ylabel("NW (fast-spiking) fraction")
    axes[2].set_title("NW Fraction by Layer")
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "figures" / "layer5_gaba_nw.png", dpi=150)
    plt.savefig(OUTPUT_DIR / "figures" / "layer5_gaba_nw.svg")

print(f"\nResults saved to {out_path}")

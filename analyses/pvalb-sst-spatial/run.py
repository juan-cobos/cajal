"""
Hypothesis H12: Pvalb interneurons cluster more tightly in spatial coordinates
than Sst or Vip interneurons within PFC regions, reflecting Pvalb's role in
perisomatic inhibition of nearby pyramidal cells vs Sst/Vip's dendritic
inhibition at longer range.

Branch: pvalb-sst-spatial
Datasets: merfish
"""

import json
import random
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from scipy.spatial import cKDTree
from scipy.stats import mannwhitneyu, kruskal

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

K_NEIGHBORS = 15


def load_merfish_pfc() -> pl.DataFrame:
    df = pl.read_csv(
        DATA_PATH / "merfish" / "cell_metadata_with_parcellation_annotation.csv"
    )
    df = df.filter(pl.col("parcellation_substructure").is_in(PFC_SUBREGIONS))
    return df


def classify_interneuron(subclass: str) -> str | None:
    if "Pvalb" in subclass:
        return "Pvalb"
    elif "Sst" in subclass:
        return "Sst"
    elif "Vip" in subclass:
        return "Vip"
    elif "Lamp5" in subclass or "Sncg" in subclass:
        return "Other_IN"
    return None


def mean_same_type_neighbor_fraction(
    coords: np.ndarray, type_arr: np.ndarray, target_type: str, k: int = K_NEIGHBORS
) -> np.ndarray:
    mask = type_arr == target_type
    idx = np.where(mask)[0]
    if len(idx) < k + 1:
        return np.array([])
    tree = cKDTree(coords)
    n_same_total = int(mask.sum())
    n_total = len(type_arr)
    expected_frac = (n_same_total - 1) / (n_total - 1) if n_total > 1 else 0

    enrichments = []
    for i in idx:
        _, neighbors = tree.query(coords[i], k=k + 1)
        obs_frac = np.sum(type_arr[neighbors[1:]] == target_type) / k
        enrichments.append(obs_frac / expected_frac if expected_frac > 0 else 1.0)
    return np.array(enrichments)


print("=" * 60)
print("H12: Pvalb vs Sst vs Vip Spatial Clustering")
print("=" * 60)

print("\nLoading MERFISH data...")
df = load_merfish_pfc()
print(f"Loaded {len(df)} cells")

df = df.with_columns(
    pl.col("subclass")
    .map_elements(classify_interneuron, return_dtype=pl.Utf8)
    .alias("in_type")
)
in_cells = df.filter(pl.col("in_type").is_not_null())
print(f"Interneurons: {len(in_cells)}")
print(f"  Types: {Counter(in_cells['in_type'].to_list())}")

all_pvalb_enr = []
all_sst_enr = []
all_vip_enr = []
region_results = []

for r in PFC_SUBREGIONS:
    sub = df.filter(pl.col("parcellation_substructure") == r)
    in_sub = sub.filter(pl.col("in_type").is_not_null())
    if len(in_sub) < 100:
        continue

    coords = np.column_stack([
        sub["x_reconstructed"].to_numpy(),
        sub["y_reconstructed"].to_numpy(),
        sub["z_reconstructed"].to_numpy(),
    ])
    type_arr = np.array(sub["in_type"].to_list() if "in_type" in sub.columns else [None] * len(sub))

    pvalb_enr = mean_same_type_neighbor_fraction(coords, type_arr, "Pvalb", K_NEIGHBORS)
    sst_enr = mean_same_type_neighbor_fraction(coords, type_arr, "Sst", K_NEIGHBORS)
    vip_enr = mean_same_type_neighbor_fraction(coords, type_arr, "Vip", K_NEIGHBORS)

    if len(pvalb_enr) > 0 and len(sst_enr) > 0 and len(vip_enr) > 0:
        all_pvalb_enr.extend(pvalb_enr.tolist())
        all_sst_enr.extend(sst_enr.tolist())
        all_vip_enr.extend(vip_enr.tolist())
        region_results.append({
            "region": r,
            "pvalb_enr": float(pvalb_enr.mean()),
            "sst_enr": float(sst_enr.mean()),
            "vip_enr": float(vip_enr.mean()),
            "n_pvalb": len(pvalb_enr),
            "n_sst": len(sst_enr),
            "n_vip": len(vip_enr),
        })
        print(f"  {r}: Pvalb={pvalb_enr.mean():.3f}, Sst={sst_enr.mean():.3f}, Vip={vip_enr.mean():.3f}")

all_pvalb = np.array(all_pvalb_enr)
all_sst = np.array(all_sst_enr)
all_vip = np.array(all_vip_enr)

print(f"\n=== Global Results ===")
print(f"N Pvalb: {len(all_pvalb)}, mean enrichment: {all_pvalb.mean():.3f}")
print(f"N Sst: {len(all_sst)}, mean enrichment: {all_sst.mean():.3f}")
print(f"N Vip: {len(all_vip)}, mean enrichment: {all_vip.mean():.3f}")

stat_pv_ss, p_pv_ss = mannwhitneyu(all_pvalb, all_sst, alternative="greater")
stat_pv_vip, p_pv_vip = mannwhitneyu(all_pvalb, all_vip, alternative="greater")

all_three = np.concatenate([all_pvalb, all_sst, all_vip])
groups = np.concatenate([np.zeros(len(all_pvalb)), np.ones(len(all_sst)), np.full(len(all_vip), 2)])
stat_kw, p_kw = kruskal(all_pvalb, all_sst, all_vip)

print(f"\nPvalb > Sst: U={stat_pv_ss:.0f}, p={p_pv_ss:.2e}")
print(f"Pvalb > Vip: U={stat_pv_vip:.0f}, p={p_pv_vip:.2e}")
print(f"Kruskal-Wallis (3 groups): H={stat_kw:.1f}, p={p_kw:.2e}")

n_perm = 10000
obs_diff = all_pvalb.mean() - max(all_sst.mean(), all_vip.mean())
null_diffs = []
combined = np.concatenate([all_pvalb, all_sst, all_vip])
n_pv = len(all_pvalb)
n_rest = len(combined) - n_pv
for _ in range(n_perm):
    perm = np.random.permutation(combined)
    null_diffs.append(perm[:n_pv].mean() - perm[n_pv:].mean())
null_diffs = np.array(null_diffs)
p_perm = float(np.mean(null_diffs >= obs_diff))
print(f"Permutation p (Pvalb > max(Sst,Vip)): {p_perm:.4f}")

status = "CONFIRMED" if (p_pv_ss < 0.05 and p_pv_vip < 0.05) else "REFUTED"

results = {
    "hypothesis_id": "H12",
    "status": status,
    "pvalb_mean_enrichment": float(all_pvalb.mean()),
    "sst_mean_enrichment": float(all_sst.mean()),
    "vip_mean_enrichment": float(all_vip.mean()),
    "pvalb_vs_sst_p": float(p_pv_ss),
    "pvalb_vs_vip_p": float(p_pv_vip),
    "kruskal_p": float(p_kw),
    "permutation_p": float(p_perm),
    "n_pvalb": len(all_pvalb),
    "n_sst": len(all_sst),
    "n_vip": len(all_vip),
    "n_regions": len(region_results),
    "region_details": region_results,
    "notes": f"Pvalb={all_pvalb.mean():.3f} > Sst={all_sst.mean():.3f} > Vip={all_vip.mean():.3f}; pv>ss p={p_pv_ss:.2e}; {status}"
}

print("\n" + json.dumps(results, indent=2))

out_path = OUTPUT_DIR / "results.json"
out_path.write_text(json.dumps(results, indent=2))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

data = [all_pvalb, all_sst, all_vip]
labels = ["Pvalb", "Sst", "Vip"]
colors = ["#DD8452", "#55A868", "#C44E52"]

bp = axes[0].boxplot(data, tick_labels=labels, patch_artist=True)
for i, box in enumerate(bp["boxes"]):
    box.set_facecolor(colors[i])
    box.set_alpha(0.7)
axes[0].set_ylabel("Same-type spatial enrichment ratio")
axes[0].set_title(f"Interneuron Spatial Clustering\nPvalb > Sst p={p_pv_ss:.2e}")
axes[0].axhline(1.0, color="black", linestyle="--", alpha=0.5)

reg_names = [r["region"] for r in region_results]
x_pos = np.arange(len(reg_names))
width = 0.25
axes[1].bar(x_pos - width, [r["pvalb_enr"] for r in region_results], width, label="Pvalb", color=colors[0], edgecolor="black")
axes[1].bar(x_pos, [r["sst_enr"] for r in region_results], width, label="Sst", color=colors[1], edgecolor="black")
axes[1].bar(x_pos + width, [r["vip_enr"] for r in region_results], width, label="Vip", color=colors[2], edgecolor="black")
axes[1].set_xticks(x_pos)
axes[1].set_xticklabels(reg_names, rotation=45, ha="right", fontsize=8)
axes[1].set_ylabel("Same-type enrichment ratio")
axes[1].set_title("Spatial Enrichment by Region")
axes[1].legend()
axes[1].axhline(1.0, color="black", linestyle="--", alpha=0.5)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "figures" / "pvalb_sst_spatial.png", dpi=150)
plt.savefig(OUTPUT_DIR / "figures" / "pvalb_sst_spatial.svg")

print(f"\nResults saved to {out_path}")

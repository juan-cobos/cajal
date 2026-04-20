"""
Hypothesis H04: GABAergic interneurons in PFC show lower same-subclass spatial
enrichment than glutamatergic neurons, reflecting greater intermixing of
interneuron subtypes (Pvalb, Sst, Vip) compared to laminar-organized
glutamatergic subtypes.
Branch: gaba-glut-spatial
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
from scipy.stats import mannwhitneyu

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

DATA_PATH = Path(__file__).parent.parent.parent / "data"
OUTPUT_DIR = Path(__file__).parent

PFC_REGIONS = [
    "PL5", "PL2/3", "PL6a",
    "ILA5", "ILA6a",
    "ORBvl5", "ORBl5", "ORBm5", "ORBm6a",
    "ORBl2/3", "ORBvl2/3",
    "ORBl6a", "ORBvl6a",
]

K_NEIGHBORS = 20


def load_merfish_pfc() -> pl.DataFrame:
    df = pl.read_csv(
        DATA_PATH / "merfish" / "cell_metadata_with_parcellation_annotation.csv"
    )
    pfc_pat = "|".join(PFC_REGIONS)
    df = df.filter(
        df["parcellation_substructure"].str.contains(pfc_pat)
    )
    return df


def compute_subclass_enrichment(
    coords: np.ndarray,
    subclass_arr: np.ndarray,
    neurotransmitter: list[str],
    target_nt: str,
    k: int = K_NEIGHBORS,
) -> np.ndarray:
    is_target = np.array([1 if g == target_nt else 0 for g in neurotransmitter])
    target_idx = np.where(is_target == 1)[0]
    if len(target_idx) < 20:
        return np.array([])

    tree = cKDTree(coords)
    n_total = len(coords)
    subclass_counts = Counter(subclass_arr)

    enrichments = []
    for i in target_idx:
        my_sub = subclass_arr[i]
        n_same = subclass_counts[my_sub]
        expected_frac = (n_same - 1) / (n_total - 1)
        _, neighbors = tree.query(coords[i], k=k + 1)
        observed_frac = sum(1 for j in neighbors[1:] if subclass_arr[j] == my_sub) / k
        if expected_frac > 0:
            enrichments.append(observed_frac / expected_frac)
        else:
            enrichments.append(1.0)

    return np.array(enrichments)


print("Loading MERFISH data...")
df = load_merfish_pfc()
print(f"Loaded {len(df)} cells from PFC regions")

all_gaba_enrichment = []
all_glut_enrichment = []
region_results = []

for r in PFC_REGIONS:
    sub = df.filter(pl.col("parcellation_substructure") == r)
    if len(sub) < 200:
        continue

    x = sub["x_reconstructed"].to_numpy()
    y = sub["y_reconstructed"].to_numpy()
    z = sub["z_reconstructed"].to_numpy()
    coords = np.column_stack([x, y, z])
    subclass_arr = np.array(sub["subclass"].to_list())
    nt = sub["neurotransmitter"].to_list()

    gaba_enr = compute_subclass_enrichment(coords, subclass_arr, nt, "GABA")
    glut_enr = compute_subclass_enrichment(coords, subclass_arr, nt, "Glut")

    if len(gaba_enr) == 0 or len(glut_enr) == 0:
        continue

    all_gaba_enrichment.extend(gaba_enr.tolist())
    all_glut_enrichment.extend(glut_enr.tolist())

    region_results.append({
        "region": r,
        "gaba_enrichment": float(gaba_enr.mean()),
        "glut_enrichment": float(glut_enr.mean()),
        "n_gaba": len(gaba_enr),
        "n_glut": len(glut_enr),
    })

    print(
        f"{r}: GABA enrichment={gaba_enr.mean():.3f}, "
        f"Glut enrichment={glut_enr.mean():.3f}, "
        f"diff={gaba_enr.mean() - glut_enr.mean():.3f}"
    )

all_gaba = np.array(all_gaba_enrichment)
all_glut = np.array(all_glut_enrichment)

n_gaba = len(all_gaba)
n_glut = len(all_glut)

if n_gaba < 20 or n_glut < 20:
    print("Insufficient data for analysis")
    results = {
        "hypothesis_id": "H04",
        "status": "INCONCLUSIVE",
        "statistic": None,
        "p_value": None,
        "effect_size": None,
        "n_cells": n_gaba + n_glut,
        "notes": "Insufficient GABA or Glut cells for analysis"
    }
else:
    stat, p_value = mannwhitneyu(all_gaba, all_glut, alternative="less")

    effect_size = all_gaba.mean() - all_glut.mean()

    print(f"\n=== Global Results ===")
    print(f"N GABA cells: {n_gaba}")
    print(f"N Glut cells: {n_glut}")
    print(f"Mean GABA subclass enrichment: {all_gaba.mean():.3f}")
    print(f"Mean Glut subclass enrichment: {all_glut.mean():.3f}")
    print(f"Effect size (GABA - Glut): {effect_size:.3f}")
    print(f"Mann-Whitney U (GABA < Glut): {stat:.0f}")
    print(f"P-value: {p_value:.2e}")

    status = "CONFIRMED" if p_value < 0.05 and effect_size < 0 else "REFUTED"

    results = {
        "hypothesis_id": "H04",
        "status": status,
        "statistic": float(stat),
        "p_value": float(p_value),
        "effect_size": float(effect_size),
        "n_cells": n_gaba + n_glut,
        "n_gaba": n_gaba,
        "n_glut": n_glut,
        "gaba_mean_enrichment": float(all_gaba.mean()),
        "glut_mean_enrichment": float(all_glut.mean()),
        "notes": f"GABA enrichment ({all_gaba.mean():.3f}) < Glut ({all_glut.mean():.3f}): {status}"
    }

print("\n" + json.dumps(results, indent=2))

out_path = OUTPUT_DIR / "results.json"
out_path.write_text(json.dumps(results, indent=2))

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

reg_names = [r["region"] for r in region_results]
gaba_means = [r["gaba_enrichment"] for r in region_results]
glut_means = [r["glut_enrichment"] for r in region_results]

x_pos = np.arange(len(reg_names))
width = 0.35

axes[0].bar(x_pos - width / 2, gaba_means, width, label="GABA", color="#DD8452", edgecolor="black")
axes[0].bar(x_pos + width / 2, glut_means, width, label="Glut", color="#4C72B0", edgecolor="black")
axes[0].set_xticks(x_pos)
axes[0].set_xticklabels(reg_names, rotation=45, ha="right", fontsize=8)
axes[0].set_ylabel("Same-subclass enrichment ratio")
axes[0].set_title("Subclass Spatial Enrichment\nby Neurotransmitter Type")
axes[0].legend()
axes[0].axhline(1.0, color="black", linestyle="--", alpha=0.5, label="Expected")

data_for_box = [all_gaba, all_glut]
bp = axes[1].boxplot(data_for_box, tick_labels=["GABA", "Glut"], patch_artist=True)
bp["boxes"][0].set_facecolor("#DD8452")
bp["boxes"][0].set_alpha(0.7)
bp["boxes"][1].set_facecolor("#4C72B0")
bp["boxes"][1].set_alpha(0.7)
axes[1].set_ylabel("Same-subclass enrichment ratio")
axes[1].set_title("Global Subclass Enrichment\n(GABA < Glut)")
axes[1].axhline(1.0, color="black", linestyle="--", alpha=0.5)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "figures" / "gaba_glut_spatial_enrichment.png", dpi=150)
plt.savefig(OUTPUT_DIR / "figures" / "gaba_glut_spatial_enrichment.svg")

print(f"\nResults saved to {out_path}")
print("Figures saved to figures/")

"""
Hypothesis H14: scRNA-seq captures greater transcriptomic cluster diversity
(higher Shannon entropy) than MERFISH in shared PFC parent regions, because
scRNA detects more fine-grained cell states (including rare subtypes) due to
full transcriptome coverage vs MERFISH's targeted gene panel.

Branch: scrna-merfish-entropy
Datasets: scrna, merfish
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

PFC_PARENTS = ["PL", "ILA", "ORBl", "ORBm", "ORBvl", "ACAd", "ACAv", "FRP"]

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

SCRNA_REGION_MAP = {
    "PL-ILA-ORB": ["PL", "ILA", "ORBl", "ORBm", "ORBvl"],
    "ACA": ["ACAd", "ACAv"],
    "MO-FRP": ["FRP"],
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
    pfc_pat = "|".join(PFC_PARENTS)
    df = df.filter(pl.col("parcellation_structure").str.contains(pfc_pat))
    return df


def load_scrna_pfc() -> pl.DataFrame:
    df = pl.read_csv(DATA_PATH / "scrna" / "cell_metadata.csv")
    pfc_regions = list(SCRNA_REGION_MAP.keys())
    df = df.filter(pl.col("region_of_interest_acronym").is_in(pfc_regions))
    return df


print("=" * 60)
print("H14: scRNA vs MERFISH Cluster Entropy in PFC")
print("=" * 60)

print("\n[1] Loading MERFISH data...")
merfish = load_merfish_pfc()
print(f"  Loaded {len(merfish)} cells")

merfish = merfish.with_columns(
    pl.col("parcellation_substructure")
    .map_elements(lambda s: SUBSTRUCTURE_TO_PARENT.get(s, None), return_dtype=pl.Utf8)
    .alias("parent_region")
)
merfish = merfish.filter(pl.col("parent_region").is_not_null())

merfish_entropy = {}
for parent in PFC_PARENTS:
    sub = merfish.filter(pl.col("parent_region") == parent)
    if len(sub) < 100:
        continue
    subclass_counts = Counter(sub["subclass"].to_list())
    ent = shannon_entropy(subclass_counts)
    merfish_entropy[parent] = {"entropy": ent, "n_cells": len(sub), "n_clusters": len(subclass_counts)}
    print(f"  {parent}: entropy={ent:.3f}, n={len(sub)}, subclasses={len(subclass_counts)}")

print("\n[2] Loading scRNA data...")
scrna = load_scrna_pfc()
print(f"  Loaded {len(scrna)} cells")

scrna_entropy = {}
for roi, parents in SCRNA_REGION_MAP.items():
    sub = scrna.filter(pl.col("region_of_interest_acronym") == roi)
    if len(sub) < 100:
        continue
    cluster_counts = Counter(sub["cluster_alias"].to_list())
    ent = shannon_entropy(cluster_counts)
    for parent in parents:
        scrna_entropy[parent] = {
            "entropy": ent,
            "n_cells": len(sub),
            "n_clusters": len(cluster_counts),
            "roi": roi,
        }
    print(f"  {roi}: entropy={ent:.3f}, n={len(sub)}, clusters={len(cluster_counts)} (maps to {parents})")

common = sorted(set(merfish_entropy.keys()) & set(scrna_entropy.keys()))
print(f"\n[3] Common parent regions: {common}")

if len(common) < 2:
    results = {"hypothesis_id": "H14", "status": "INCONCLUSIVE", "n_common": len(common)}
else:
    merfish_ents = np.array([merfish_entropy[r]["entropy"] for r in common])
    scrna_ents = np.array([scrna_entropy[r]["entropy"] for r in common])
    diff = scrna_ents - merfish_ents

    print(f"\n=== Entropy Comparison ===")
    for i, r in enumerate(common):
        print(f"  {r}: scRNA={scrna_ents[i]:.3f}, MERFISH={merfish_ents[i]:.3f}, diff={diff[i]:+.3f}")

    print(f"\n  Mean scRNA entropy: {scrna_ents.mean():.3f}")
    print(f"  Mean MERFISH entropy: {merfish_ents.mean():.3f}")
    print(f"  Mean diff (scRNA - MERFISH): {diff.mean():.3f}")

    n_greater_scrna = int(np.sum(diff > 0))
    print(f"  Regions where scRNA > MERFISH: {n_greater_scrna}/{len(common)}")

    if len(common) >= 3:
        stat, p_val = wilcoxon(scrna_ents, merfish_ents, alternative="greater")
        print(f"  Wilcoxon signed-rank (scRNA > MERFISH): stat={stat:.1f}, p={p_val:.4f}")
    else:
        p_val = 1.0
        stat = None

    status = "CONFIRMED" if (p_val < 0.05 and diff.mean() > 0) else "REFUTED"

    region_details = []
    for r in common:
        region_details.append({
            "region": r,
            "scrna_entropy": round(scrna_entropy[r]["entropy"], 4),
            "merfish_entropy": round(merfish_entropy[r]["entropy"], 4),
            "scrna_n_cells": scrna_entropy[r]["n_cells"],
            "merfish_n_cells": merfish_entropy[r]["n_cells"],
            "scrna_n_clusters": scrna_entropy[r]["n_clusters"],
            "merfish_n_clusters": merfish_entropy[r]["n_clusters"],
            "scrna_roi": scrna_entropy[r].get("roi", ""),
        })

    results = {
        "hypothesis_id": "H14",
        "status": status,
        "wilcoxon_p": float(p_val),
        "mean_scrna_entropy": float(scrna_ents.mean()),
        "mean_merfish_entropy": float(merfish_ents.mean()),
        "mean_diff": float(diff.mean()),
        "n_greater_scrna": n_greater_scrna,
        "n_regions": len(common),
        "region_details": region_details,
        "notes": f"scRNA mean={scrna_ents.mean():.3f} vs MERFISH mean={merfish_ents.mean():.3f}, diff={diff.mean():.3f}; p={p_val:.4f}; {status}"
    }

print("\n" + json.dumps(results, indent=2))

out_path = OUTPUT_DIR / "results.json"
out_path.write_text(json.dumps(results, indent=2))

if len(common) >= 2:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    x_pos = np.arange(len(common))
    width = 0.35
    axes[0].bar(x_pos - width/2, [scrna_entropy[r]["entropy"] for r in common], width, label="scRNA", color="#4C72B0", edgecolor="black")
    axes[0].bar(x_pos + width/2, [merfish_entropy[r]["entropy"] for r in common], width, label="MERFISH", color="#55A868", edgecolor="black")
    axes[0].set_xticks(x_pos)
    axes[0].set_xticklabels(common, rotation=45, ha="right")
    axes[0].set_ylabel("Shannon entropy (bits)")
    axes[0].set_title("Cluster Entropy by Modality")
    axes[0].legend()

    axes[1].scatter(merfish_ents, scrna_ents, s=120, zorder=5, edgecolors="black", linewidths=0.8)
    for i, r in enumerate(common):
        axes[1].annotate(r, (merfish_ents[i], scrna_ents[i]), textcoords="offset points", xytext=(6, 4), fontsize=9)
    lims = [min(merfish_ents.min(), scrna_ents.min()) - 0.2, max(merfish_ents.max(), scrna_ents.max()) + 0.2]
    axes[1].plot(lims, lims, "k--", alpha=0.3, label="y = x")
    axes[1].set_xlabel("MERFISH entropy (bits)")
    axes[1].set_ylabel("scRNA entropy (bits)")
    axes[1].set_title("Entropy: scRNA vs MERFISH")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "figures" / "scrna_merfish_entropy.png", dpi=150)
    plt.savefig(OUTPUT_DIR / "figures" / "scrna_merfish_entropy.svg")

print(f"\nResults saved to {out_path}")

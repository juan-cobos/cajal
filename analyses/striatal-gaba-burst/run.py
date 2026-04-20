"""
Hypothesis H10: PFC subregions with stronger striatal projections have higher
GABAergic cell proportions (MERFISH) and higher burst indices (neural_activity),
reflecting the role of striatal-projecting PFC regions in inhibitory control.

Three-way cross-modal: connectivity → transcriptomics → electrophysiology

Branch: striatal-gaba-burst
Datasets: connectivity, merfish, neural_activity
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
]

STRIATAL_ACRONYMS = [
    "STR", "STRd", "STRv", "CP", "ACB", "LS", "LSX", "BST", "SI",
]


def load_striatal_ids() -> set[int]:
    from src.structures import ACRONYM_TO_ID
    ids = set()
    for acr in STRIATAL_ACRONYMS:
        if acr in ACRONYM_TO_ID:
            ids.add(ACRONYM_TO_ID[acr])
    return ids


def load_connectivity_pfc() -> pl.DataFrame:
    cache = DATA_PATH / "connectivity"
    frames = []
    for region in PFC_SUBREGIONS:
        for f in sorted(cache.glob(f"{region}_*.parquet")):
            frames.append(pl.read_parquet(f).with_columns(pl.lit(region).alias("input_region")))
    if not frames:
        return pl.DataFrame()
    return pl.concat(frames)


def load_merfish_pfc() -> pl.DataFrame:
    df = pl.read_csv(
        DATA_PATH / "merfish" / "cell_metadata_with_parcellation_annotation.csv"
    )
    df = df.filter(pl.col("parcellation_substructure").is_in(PFC_SUBREGIONS))
    return df


def load_neural_activity_pfc() -> pl.DataFrame:
    df = pl.read_csv(DATA_PATH / "pfcmap" / "units.csv")
    df = df.filter(pl.col("region").is_in(PFC_SUBREGIONS))
    df = df.filter(pl.col("B_mean").is_not_null())
    df = df.filter(pl.col("rate_mean").is_not_null())
    return df


print("=" * 60)
print("H10: Striatal Projection → GABA Fraction → Burst Index")
print("=" * 60)

print("\n[1] Loading connectivity data...")
conn = load_connectivity_pfc()
print(f"  Loaded {len(conn)} rows")

stri_ids = load_striatal_ids()
print(f"  Striatal structure IDs: {len(stri_ids)}")

conn_proj = conn.filter(pl.col("is_injection") == False)
conn_stri = conn_proj.filter(pl.col("structure_id").is_in(list(stri_ids)))
print(f"  Striatal projection rows: {len(conn_stri)}")

stri_strength = {}
for region in PFC_SUBREGIONS:
    sub = conn_stri.filter(pl.col("input_region") == region)
    if len(sub) == 0:
        continue
    mean_npv = float(sub["normalized_projection_volume"].mean())
    n_exp = sub["section_data_set_id"].n_unique()
    stri_strength[region] = {"mean_npv": mean_npv, "n_experiments": n_exp}
    print(f"  {region}: striatal NPV={mean_npv:.4f}, n_exp={n_exp}")

print("\n[2] Loading MERFISH data...")
merfish = load_merfish_pfc()
print(f"  Loaded {len(merfish)} cells")

gaba_fraction = {}
for region in PFC_SUBREGIONS:
    sub = merfish.filter(pl.col("parcellation_substructure") == region)
    if len(sub) == 0:
        continue
    nt_counts = Counter(sub["neurotransmitter"].to_list())
    n_gaba = nt_counts.get("GABA", 0)
    n_glut = nt_counts.get("Glut", 0)
    n_total_nt = n_gaba + n_glut
    if n_total_nt == 0:
        continue
    gaba_frac = n_gaba / n_total_nt
    gaba_fraction[region] = {
        "gaba_fraction": gaba_frac,
        "n_gaba": n_gaba,
        "n_glut": n_glut,
        "n_cells": len(sub),
    }
    print(f"  {region}: GABA frac={gaba_frac:.3f} (GABA={n_gaba}, Glut={n_glut})")

print("\n[3] Loading neural activity data...")
neural = load_neural_activity_pfc()
print(f"  Loaded {len(neural)} neurons")

burst_per_region = {}
for region in PFC_SUBREGIONS:
    sub = neural.filter(pl.col("region") == region)
    if len(sub) < 10:
        continue
    mean_burst = float(sub["B_mean"].mean())
    mean_rate = float(sub["rate_mean"].mean())
    burst_per_region[region] = {
        "mean_burst": mean_burst,
        "mean_rate": mean_rate,
        "n_neurons": len(sub),
    }
    print(f"  {region}: burst={mean_burst:.4f}, rate={mean_rate:.4f}, n={len(sub)}")

common = sorted(set(stri_strength.keys()) & set(gaba_fraction.keys()) & set(burst_per_region.keys()))
print(f"\n[4] Common regions: {common}")

if len(common) < 4:
    results = {
        "hypothesis_id": "H10",
        "status": "INCONCLUSIVE",
        "n_regions": len(common),
        "notes": "Too few common regions for correlation"
    }
else:
    npv_vals = np.array([stri_strength[r]["mean_npv"] for r in common])
    gaba_vals = np.array([gaba_fraction[r]["gaba_fraction"] for r in common])
    burst_vals = np.array([burst_per_region[r]["mean_burst"] for r in common])
    rate_vals = np.array([burst_per_region[r]["mean_rate"] for r in common])

    rho_npv_gaba, p_npv_gaba = spearmanr(npv_vals, gaba_vals)
    rho_gaba_burst, p_gaba_burst = spearmanr(gaba_vals, burst_vals)
    rho_npv_burst, p_npv_burst = spearmanr(npv_vals, burst_vals)
    rho_npv_rate, p_npv_rate = spearmanr(npv_vals, rate_vals)
    rho_gaba_rate, p_gaba_rate = spearmanr(gaba_vals, rate_vals)

    print(f"\n=== Correlation Matrix ===")
    print(f"  Striatal NPV vs GABA frac:   rho={rho_npv_gaba:.4f}, p={p_npv_gaba:.4f}")
    print(f"  GABA frac vs Burst index:     rho={rho_gaba_burst:.4f}, p={p_gaba_burst:.4f}")
    print(f"  Striatal NPV vs Burst index:  rho={rho_npv_burst:.4f}, p={p_npv_burst:.4f}")
    print(f"  Striatal NPV vs Mean rate:    rho={rho_npv_rate:.4f}, p={p_npv_rate:.4f}")
    print(f"  GABA frac vs Mean rate:       rho={rho_gaba_rate:.4f}, p={p_gaba_rate:.4f}")

    chain_supported = (rho_npv_gaba > 0 and p_npv_gaba < 0.1 and
                       rho_gaba_burst > 0 and p_gaba_burst < 0.1)

    n_perm = 10000
    null_npv_gaba = []
    null_gaba_burst = []
    for _ in range(n_perm):
        perm_gaba = np.random.permutation(gaba_vals)
        r1, _ = spearmanr(npv_vals, perm_gaba)
        r2, _ = spearmanr(perm_gaba, burst_vals)
        null_npv_gaba.append(r1)
        null_gaba_burst.append(r2)
    null_npv_gaba = np.array(null_npv_gaba)
    null_gaba_burst = np.array(null_gaba_burst)
    p_perm_npv_gaba = float(np.mean(null_npv_gaba >= rho_npv_gaba))
    p_perm_gaba_burst = float(np.mean(null_gaba_burst >= rho_gaba_burst))

    print(f"\n  Permutation p (NPV→GABA): {p_perm_npv_gaba:.4f}")
    print(f"  Permutation p (GABA→burst): {p_perm_gaba_burst:.4f}")

    status = "CONFIRMED" if chain_supported else "REFUTED"

    region_details = []
    for r in common:
        region_details.append({
            "region": r,
            "striatal_NPV": round(stri_strength[r]["mean_npv"], 4),
            "gaba_fraction": round(gaba_fraction[r]["gaba_fraction"], 4),
            "mean_burst": round(burst_per_region[r]["mean_burst"], 4),
            "mean_rate": round(burst_per_region[r]["mean_rate"], 4),
            "n_merfish": gaba_fraction[r]["n_cells"],
            "n_neurons": burst_per_region[r]["n_neurons"],
        })

    results = {
        "hypothesis_id": "H10",
        "status": status,
        "rho_npv_gaba": float(rho_npv_gaba),
        "p_npv_gaba": float(p_npv_gaba),
        "rho_gaba_burst": float(rho_gaba_burst),
        "p_gaba_burst": float(p_gaba_burst),
        "rho_npv_burst": float(rho_npv_burst),
        "p_npv_burst": float(p_npv_burst),
        "rho_npv_rate": float(rho_npv_rate),
        "p_npv_rate": float(p_npv_rate),
        "rho_gaba_rate": float(rho_gaba_rate),
        "p_gaba_rate": float(p_gaba_rate),
        "p_perm_npv_gaba": float(p_perm_npv_gaba),
        "p_perm_gaba_burst": float(p_perm_gaba_burst),
        "n_regions": len(common),
        "chain_supported": bool(chain_supported),
        "region_details": region_details,
        "notes": f"NPV→GABA: rho={rho_npv_gaba:.3f} p={p_npv_gaba:.3f}; GABA→burst: rho={rho_gaba_burst:.3f} p={p_gaba_burst:.3f}; {status}"
    }

print("\n" + json.dumps(results, indent=2))

out_path = OUTPUT_DIR / "results.json"
out_path.write_text(json.dumps(results, indent=2))

if len(common) >= 4:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    npv_vals = np.array([stri_strength[r]["mean_npv"] for r in common])
    gaba_vals = np.array([gaba_fraction[r]["gaba_fraction"] for r in common])
    burst_vals = np.array([burst_per_region[r]["mean_burst"] for r in common])

    def scatter_with_fit(ax, x, y, xlabel, ylabel, title, rho, p):
        ax.scatter(x, y, s=120, zorder=5, edgecolors="black", linewidths=0.8)
        for i, r in enumerate(common):
            ax.annotate(r, (x[i], y[i]), textcoords="offset points", xytext=(6, 4), fontsize=7)
        m, b = np.polyfit(x, y, 1)
        x_line = np.linspace(x.min() * 0.9, x.max() * 1.1, 100)
        ax.plot(x_line, m * x_line + b, "--", color="gray", alpha=0.7)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(f"{title}\nρ = {rho:.3f}, p = {p:.3f}")

    scatter_with_fit(
        axes[0], npv_vals, gaba_vals,
        "Striatal projection (NPV)", "GABA fraction",
        "Connectivity → Transcriptomics", rho_npv_gaba, p_npv_gaba
    )
    scatter_with_fit(
        axes[1], gaba_vals, burst_vals,
        "GABA fraction", "Mean burst index",
        "Transcriptomics → Electrophysiology", rho_gaba_burst, p_gaba_burst
    )
    scatter_with_fit(
        axes[2], npv_vals, burst_vals,
        "Striatal projection (NPV)", "Mean burst index",
        "Connectivity → Electrophysiology", rho_npv_burst, p_npv_burst
    )

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "figures" / "striatal_gaba_burst.png", dpi=150)
    plt.savefig(OUTPUT_DIR / "figures" / "striatal_gaba_burst.svg")

print(f"\nResults saved to {out_path}")
print("Figures saved to figures/")

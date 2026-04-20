"""
Hypothesis H09: PFC subregions with stronger thalamic projections (PFC→TH
normalized_projection_volume) have lower proportions of narrow-width (fast-spiking)
neurons, because corticothalamic projection neurons are predominantly regular-spiking.

Branch: connectivity-thal-nw
Datasets: connectivity, neural_activity
"""

import json
import random
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

THALAMIC_ACRONYMS = [
    "AM", "AV", "CL", "CM", "GENd", "GENv", "IMD", "IPL", "LD",
    "LGd", "LGv", "LP", "MD", "MG", "MGd", "MGv", "PCN", "PF",
    "PIL", "PO", "PVT", "PoT", "RE", "RT", "SG", "SPA", "TH",
    "VAL", "VL", "VM", "VPL", "VPM", "ZI",
]


def load_thalamic_ids() -> set[int]:
    from src.structures import ACRONYM_TO_ID
    ids = set()
    for acr in THALAMIC_ACRONYMS:
        if acr in ACRONYM_TO_ID:
            ids.add(ACRONYM_TO_ID[acr])
    return ids


def load_connectivity_pfc() -> pl.DataFrame:
    cache = DATA_PATH / "connectivity"
    frames = []
    for region in PFC_SUBREGIONS:
        region_files = sorted(cache.glob(f"{region}_*.parquet"))
        for f in region_files:
            df = pl.read_parquet(f).with_columns(pl.lit(region).alias("input_region"))
            frames.append(df)
    if not frames:
        return pl.DataFrame()
    return pl.concat(frames)


def load_neural_activity_pfc() -> pl.DataFrame:
    df = pl.read_csv(DATA_PATH / "pfcmap" / "units.csv")
    df = df.filter(pl.col("region").is_in(PFC_SUBREGIONS))
    df = df.filter(pl.col("utype").is_in(["ww", "nw"]))
    df = df.filter(pl.col("rate_mean").is_not_null())
    return df


print("Loading connectivity data...")
conn = load_connectivity_pfc()
print(f"Loaded {len(conn)} connectivity rows")

thal_ids = load_thalamic_ids()
print(f"Thalamic structure IDs: {len(thal_ids)}")

conn_proj = conn.filter(pl.col("is_injection") == False)
conn_thal = conn_proj.filter(pl.col("structure_id").is_in(list(thal_ids)))
print(f"Thalamic projection rows: {len(conn_thal)}")

thal_strength = {}
for region in PFC_SUBREGIONS:
    sub = conn_thal.filter(pl.col("input_region") == region)
    if len(sub) == 0:
        print(f"  {region}: no thalamic projection data")
        continue
    mean_npv = float(sub["normalized_projection_volume"].mean())
    mean_pd = float(sub["projection_density"].mean())
    n_exp = sub["section_data_set_id"].n_unique()
    thal_strength[region] = {
        "mean_npv": mean_npv,
        "mean_projection_density": mean_pd,
        "n_experiments": n_exp,
        "n_rows": len(sub),
    }
    print(f"  {region}: mean_NPV={mean_npv:.4f}, mean_PD={mean_pd:.4f}, n_exp={n_exp}")

print("\nLoading neural activity data...")
neural = load_neural_activity_pfc()
print(f"Loaded {len(neural)} neurons")

nw_fraction = {}
for region in PFC_SUBREGIONS:
    sub = neural.filter(pl.col("region") == region)
    if len(sub) == 0:
        continue
    n_total = len(sub)
    n_nw = len(sub.filter(pl.col("utype") == "nw"))
    n_ww = len(sub.filter(pl.col("utype") == "ww"))
    nw_frac = n_nw / n_total if n_total > 0 else 0
    nw_fraction[region] = {
        "nw_fraction": nw_frac,
        "n_nw": n_nw,
        "n_ww": n_ww,
        "n_total": n_total,
        "mean_rate": float(sub["rate_mean"].mean()),
    }
    print(f"  {region}: NW frac={nw_frac:.3f}, n={n_total} (NW={n_nw}, WW={n_ww})")

common_regions = sorted(set(thal_strength.keys()) & set(nw_fraction.keys()))
print(f"\nCommon regions: {common_regions}")

if len(common_regions) < 4:
    print("Too few common regions for correlation")
    results = {
        "hypothesis_id": "H09",
        "status": "INCONCLUSIVE",
        "notes": f"Only {len(common_regions)} common regions"
    }
else:
    npv_vals = np.array([thal_strength[r]["mean_npv"] for r in common_regions])
    nw_vals = np.array([nw_fraction[r]["nw_fraction"] for r in common_regions])
    rate_vals = np.array([nw_fraction[r]["mean_rate"] for r in common_regions])

    rho_npv_nw, p_npv_nw = spearmanr(npv_vals, nw_vals)
    rho_npv_rate, p_npv_rate = spearmanr(npv_vals, rate_vals)

    print(f"\n=== Spearman Correlations ===")
    print(f"Thalamic NPV vs NW fraction: rho={rho_npv_nw:.4f}, p={p_npv_nw:.4f}")
    print(f"Thalamic NPV vs mean rate: rho={rho_npv_rate:.4f}, p={p_npv_rate:.4f}")

    n_perm = 10000
    null_rhos = []
    for _ in range(n_perm):
        perm = np.random.permutation(nw_vals)
        r_null, _ = spearmanr(npv_vals, perm)
        null_rhos.append(r_null)
    null_rhos = np.array(null_rhos)
    p_perm = float(np.mean(null_rhos >= rho_npv_nw))
    print(f"Permutation p (NW frac vs NPV, rho >= observed): {p_perm:.4f}")

    pearson_r = float(np.corrcoef(npv_vals, nw_vals)[0, 1])
    print(f"Pearson r: {pearson_r:.4f}")

    status = "CONFIRMED" if (p_npv_nw < 0.05 and rho_npv_nw < 0) else "REFUTED"

    region_details = []
    for r in common_regions:
        region_details.append({
            "region": r,
            "thalamic_NPV": round(thal_strength[r]["mean_npv"], 4),
            "thalamic_PD": round(thal_strength[r]["mean_projection_density"], 4),
            "n_exp": thal_strength[r]["n_experiments"],
            "nw_fraction": round(nw_fraction[r]["nw_fraction"], 4),
            "n_neurons": nw_fraction[r]["n_total"],
            "mean_rate": round(nw_fraction[r]["mean_rate"], 4),
        })

    results = {
        "hypothesis_id": "H09",
        "status": status,
        "spearman_rho": float(rho_npv_nw),
        "spearman_p": float(p_npv_nw),
        "permutation_p": float(p_perm),
        "pearson_r": float(pearson_r),
        "n_regions": len(common_regions),
        "rho_npv_rate": float(rho_npv_rate),
        "p_npv_rate": float(p_npv_rate),
        "region_details": region_details,
        "notes": f"rho={rho_npv_nw:.3f}, p={p_npv_nw:.4f}; {status}"
    }

print("\n" + json.dumps(results, indent=2))

out_path = OUTPUT_DIR / "results.json"
out_path.write_text(json.dumps(results, indent=2))

if len(common_regions) >= 4:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    npv_vals = np.array([thal_strength[r]["mean_npv"] for r in common_regions])
    nw_vals = np.array([nw_fraction[r]["nw_fraction"] for r in common_regions])

    axes[0].scatter(npv_vals, nw_vals, s=120, zorder=5, edgecolors="black", linewidths=0.8)
    for i, r in enumerate(common_regions):
        axes[0].annotate(
            r, (npv_vals[i], nw_vals[i]),
            textcoords="offset points", xytext=(8, 5), fontsize=8
        )
    m, b = np.polyfit(npv_vals, nw_vals, 1)
    x_line = np.linspace(npv_vals.min() * 0.9, npv_vals.max() * 1.1, 100)
    axes[0].plot(x_line, m * x_line + b, "--", color="gray", alpha=0.7)
    axes[0].set_xlabel("Thalamic projection strength (mean NPV)")
    axes[0].set_ylabel("NW (fast-spiking) fraction")
    axes[0].set_title(
        f"Thalamic Projection vs NW Fraction\n"
        f"Spearman ρ = {rho_npv_nw:.3f}, p = {p_npv_nw:.3f}"
    )

    x_pos = np.arange(len(common_regions))
    width = 0.35
    axes[1].bar(x_pos - width / 2, npv_vals, width, label="Thalamic NPV", color="#8172B2", edgecolor="black")
    ax2 = axes[1].twinx()
    ax2.bar(x_pos + width / 2, nw_vals, width, label="NW fraction", color="#C44E52", edgecolor="black", alpha=0.8)
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels(common_regions, rotation=45, ha="right", fontsize=8)
    axes[1].set_ylabel("Thalamic projection (NPV)", color="#8172B2")
    ax2.set_ylabel("NW fraction", color="#C44E52")
    axes[1].set_title("Thalamic Projection & NW Fraction\nby PFC Subregion")
    lines1, labels1 = axes[1].get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    axes[1].legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "figures" / "connectivity_thal_nw.png", dpi=150)
    plt.savefig(OUTPUT_DIR / "figures" / "connectivity_thal_nw.svg")

print(f"\nResults saved to {out_path}")
print("Figures saved to figures/")

"""
Hypothesis H11: PFC subregions receiving stronger MD (mediodorsal) thalamic
input have higher GABAergic cell proportions, because MD is the primary
"driver" thalamic input to PFC and preferentially recruits local feedforward
inhibition via fast-spiking interneurons.

Branch: md-thal-gaba
Datasets: connectivity, merfish
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

MD_ID = 362
RE_ID = 181
PVT_ID = 149
VM_ID = 685


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


print("=" * 60)
print("H11: MD Thalamic Projection → GABAergic Fraction")
print("=" * 60)

print("\n[1] Loading connectivity data...")
conn = load_connectivity_pfc()
print(f"  Loaded {len(conn)} rows")

conn_proj = conn.filter(pl.col("is_injection") == False)

md_strength = {}
for region in PFC_SUBREGIONS:
    sub = conn_proj.filter(pl.col("input_region") == region)
    md = sub.filter(pl.col("structure_id") == MD_ID)
    re = sub.filter(pl.col("structure_id") == RE_ID)
    pvt = sub.filter(pl.col("structure_id") == PVT_ID)
    vm = sub.filter(pl.col("structure_id") == VM_ID)
    if len(md) > 0:
        md_strength[region] = {
            "md_npv": float(md["normalized_projection_volume"].mean()),
            "re_npv": float(re["normalized_projection_volume"].mean()) if len(re) > 0 else 0,
            "pvt_npv": float(pvt["normalized_projection_volume"].mean()) if len(pvt) > 0 else 0,
            "vm_npv": float(vm["normalized_projection_volume"].mean()) if len(vm) > 0 else 0,
            "n_exp": md["section_data_set_id"].n_unique(),
        }
        print(f"  {region}: MD NPV={md_strength[region]['md_npv']:.4f}, RE={md_strength[region]['re_npv']:.4f}")

print("\n[2] Loading MERFISH data...")
merfish = load_merfish_pfc()
print(f"  Loaded {len(merfish)} cells")

gaba_fraction = {}
pvalb_fraction = {}
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

    subclass_counts = Counter(sub["subclass"].to_list())
    n_pvalb = sum(v for k, v in subclass_counts.items() if "Pvalb" in k)
    n_sst = sum(v for k, v in subclass_counts.items() if "Sst" in k)

    gaba_fraction[region] = n_gaba / n_total_nt
    pvalb_fraction[region] = n_pvalb / n_total_nt if n_total_nt > 0 else 0
    print(f"  {region}: GABA={n_gaba/n_total_nt:.3f}, Pvalb={n_pvalb/n_total_nt:.4f}")

common = sorted(set(md_strength.keys()) & set(gaba_fraction.keys()))
print(f"\n[3] Common regions: {common}")

if len(common) < 4:
    results = {"hypothesis_id": "H11", "status": "INCONCLUSIVE", "n_regions": len(common)}
else:
    md_vals = np.array([md_strength[r]["md_npv"] for r in common])
    gaba_vals = np.array([gaba_fraction[r] for r in common])
    pvalb_vals = np.array([pvalb_fraction[r] for r in common])

    rho_md_gaba, p_md_gaba = spearmanr(md_vals, gaba_vals)
    rho_md_pvalb, p_md_pvalb = spearmanr(md_vals, pvalb_vals)

    print(f"\n=== Results ===")
    print(f"  MD NPV vs GABA fraction:  rho={rho_md_gaba:.4f}, p={p_md_gaba:.4f}")
    print(f"  MD NPV vs Pvalb fraction: rho={rho_md_pvalb:.4f}, p={p_md_pvalb:.4f}")

    n_perm = 10000
    null_rho = []
    for _ in range(n_perm):
        perm = np.random.permutation(gaba_vals)
        r_null, _ = spearmanr(md_vals, perm)
        null_rho.append(r_null)
    null_rho = np.array(null_rho)
    p_perm = float(np.mean(null_rho >= rho_md_gaba))
    print(f"  Permutation p (GABA, rho >= observed): {p_perm:.4f}")

    re_vals = np.array([md_strength[r]["re_npv"] for r in common])
    rho_re_gaba, p_re_gaba = spearmanr(re_vals, gaba_vals)
    print(f"  RE NPV vs GABA fraction:  rho={rho_re_gaba:.4f}, p={p_re_gaba:.4f}")

    status = "CONFIRMED" if (p_md_gaba < 0.05 and rho_md_gaba > 0) else "REFUTED"

    region_details = []
    for r in common:
        region_details.append({
            "region": r,
            "md_npv": round(md_strength[r]["md_npv"], 4),
            "gaba_fraction": round(gaba_fraction[r], 4),
            "pvalb_fraction": round(pvalb_fraction[r], 5),
        })

    results = {
        "hypothesis_id": "H11",
        "status": status,
        "spearman_rho_md_gaba": float(rho_md_gaba),
        "spearman_p_md_gaba": float(p_md_gaba),
        "permutation_p": float(p_perm),
        "spearman_rho_md_pvalb": float(rho_md_pvalb),
        "spearman_p_md_pvalb": float(p_md_pvalb),
        "spearman_rho_re_gaba": float(rho_re_gaba),
        "spearman_p_re_gaba": float(p_re_gaba),
        "n_regions": len(common),
        "region_details": region_details,
        "notes": f"MD→GABA: rho={rho_md_gaba:.3f} p={p_md_gaba:.3f}; MD→Pvalb: rho={rho_md_pvalb:.3f} p={p_md_pvalb:.3f}; {status}"
    }

print("\n" + json.dumps(results, indent=2))

out_path = OUTPUT_DIR / "results.json"
out_path.write_text(json.dumps(results, indent=2))

if len(common) >= 4:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    md_vals = np.array([md_strength[r]["md_npv"] for r in common])
    gaba_vals = np.array([gaba_fraction[r] for r in common])
    pvalb_vals = np.array([pvalb_fraction[r] for r in common])

    for ax, x, y, xlab, ylab, title, rho, p in [
        (axes[0], md_vals, gaba_vals, "MD thalamic projection (NPV)", "GABA fraction", "MD Projection → GABA Fraction", rho_md_gaba, p_md_gaba),
        (axes[1], md_vals, pvalb_vals, "MD thalamic projection (NPV)", "Pvalb fraction", "MD Projection → Pvalb Fraction", rho_md_pvalb, p_md_pvalb),
        (axes[2], re_vals, gaba_vals, "RE thalamic projection (NPV)", "GABA fraction", "RE Projection → GABA Fraction", rho_re_gaba, p_re_gaba),
    ]:
        ax.scatter(x, y, s=120, zorder=5, edgecolors="black", linewidths=0.8)
        for i, r in enumerate(common):
            ax.annotate(r, (x[i], y[i]), textcoords="offset points", xytext=(6, 4), fontsize=7)
        m, b = np.polyfit(x, y, 1)
        x_line = np.linspace(x.min() * 0.9, x.max() * 1.1, 100)
        ax.plot(x_line, m * x_line + b, "--", color="gray", alpha=0.7)
        ax.set_xlabel(xlab)
        ax.set_ylabel(ylab)
        ax.set_title(f"{title}\nρ = {rho:.3f}, p = {p:.3f}")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "figures" / "md_thal_gaba.png", dpi=150)
    plt.savefig(OUTPUT_DIR / "figures" / "md_thal_gaba.svg")

print(f"\nResults saved to {out_path}")

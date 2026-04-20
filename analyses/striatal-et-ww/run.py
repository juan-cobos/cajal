"""
Hypothesis H21: PFC subregions with stronger striatal projections have higher
L5 ET (extratelencephalic) glutamatergic fractions and higher WW (regular-
spiking) firing rates. The chain: striatal projection strength → L5 ET Glut
fraction → WW firing rate. This is restricted to WW neurons only, removing
the interneuron confound from H10/H18.

Branch: striatal-et-ww
Datasets: connectivity, merfish, neural_activity
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
ALL_REGIONS = L5_REGIONS + L6_REGIONS
PARENT_NAMES = ["PL", "ILA", "ORBl", "ORBm", "ORBvl"]

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
    for region in ALL_REGIONS:
        for f in sorted(cache.glob(f"{region}_*.parquet")):
            frames.append(pl.read_parquet(f).with_columns(pl.lit(region).alias("input_region")))
    if not frames:
        return pl.DataFrame()
    return pl.concat(frames)


def load_merfish_pfc() -> pl.DataFrame:
    df = pl.read_csv(
        DATA_PATH / "merfish" / "cell_metadata_with_parcellation_annotation.csv"
    )
    df = df.filter(pl.col("parcellation_substructure").is_in(ALL_REGIONS))
    return df


def load_neural_activity_pfc() -> pl.DataFrame:
    df = pl.read_csv(DATA_PATH / "pfcmap" / "units.csv")
    df = df.filter(pl.col("region").is_in(ALL_REGIONS))
    df = df.filter(pl.col("utype") == "ww")
    df = df.filter(pl.col("rate_mean").is_not_null())
    df = df.filter(pl.col("B_mean").is_not_null())
    return df


print("=" * 60)
print("H21: Striatal Projection → L5 ET Fraction → WW Firing Rate")
print("=" * 60)

print("\n[1] Loading connectivity data...")
conn = load_connectivity_pfc()
stri_ids = load_striatal_ids()
conn_proj = conn.filter(pl.col("is_injection") == False)
conn_stri = conn_proj.filter(pl.col("structure_id").is_in(list(stri_ids)))

stri_npv = {}
for region in ALL_REGIONS:
    sub = conn_stri.filter(pl.col("input_region") == region)
    if len(sub) == 0:
        continue
    stri_npv[region] = float(sub["normalized_projection_volume"].mean())
    print(f"  {region}: striatal NPV={stri_npv[region]:.4f}")

print("\n[2] Loading MERFISH data...")
merfish = load_merfish_pfc()
print(f"  Loaded {len(merfish)} cells")

et_fraction = {}
for region in ALL_REGIONS:
    sub = merfish.filter(pl.col("parcellation_substructure") == region)
    glut_sub = sub.filter(pl.col("neurotransmitter") == "Glut")
    if len(glut_sub) == 0:
        continue
    subclass_counts = Counter(glut_sub["subclass"].to_list())
    n_et = sum(v for k, v in subclass_counts.items() if "ET" in k)
    n_it = sum(v for k, v in subclass_counts.items() if "IT" in k)
    n_ct = sum(v for k, v in subclass_counts.items() if "CT" in k or "L6b" in k)
    n_np = sum(v for k, v in subclass_counts.items() if "NP" in k)
    n_glut = len(glut_sub)
    et_fraction[region] = {
        "et_frac": n_et / n_glut,
        "it_frac": n_it / n_glut,
        "ct_frac": n_ct / n_glut,
        "np_frac": n_np / n_glut,
        "n_glut": n_glut,
        "n_et": n_et,
    }
    print(f"  {region}: ET={n_et}/{n_glut} ({n_et/n_glut:.1%}), IT={n_it/n_glut:.1%}, CT={n_ct/n_glut:.1%}")

print("\n[3] Loading neural activity data (WW only)...")
neural = load_neural_activity_pfc()
print(f"  Loaded {len(neural)} WW neurons")

ww_rate = {}
ww_burst = {}
for region in ALL_REGIONS:
    sub = neural.filter(pl.col("region") == region)
    if len(sub) < 5:
        continue
    ww_rate[region] = float(sub["rate_mean"].mean())
    ww_burst[region] = float(sub["B_mean"].mean())
    print(f"  {region}: WW rate={ww_rate[region]:.3f}, burst={ww_burst[region]:.4f}, n={len(sub)}")

common = sorted(set(stri_npv.keys()) & set(et_fraction.keys()) & set(ww_rate.keys()))
print(f"\n[4] Common regions ({len(common)}): {common}")

npv_vals = np.array([stri_npv[r] for r in common])
et_vals = np.array([et_fraction[r]["et_frac"] for r in common])
rate_vals = np.array([ww_rate[r] for r in common])
burst_vals = np.array([ww_burst[r] for r in common])

print(f"\n{'='*60}")
print("FULL ANALYSIS (all 10 subregions)")
print(f"{'='*60}")

rho_npv_et, p_npv_et = spearmanr(npv_vals, et_vals)
rho_et_rate, p_et_rate = spearmanr(et_vals, rate_vals)
rho_npv_rate, p_npv_rate = spearmanr(npv_vals, rate_vals)
rho_npv_burst, p_npv_burst = spearmanr(npv_vals, burst_vals)
rho_et_burst, p_et_burst = spearmanr(et_vals, burst_vals)

print(f"  Striatal NPV → ET fraction:   ρ={rho_npv_et:.3f}, p={p_npv_et:.4f}")
print(f"  ET fraction → WW rate:        ρ={rho_et_rate:.3f}, p={p_et_rate:.4f}")
print(f"  Striatal NPV → WW rate:       ρ={rho_npv_rate:.3f}, p={p_npv_rate:.4f}")
print(f"  Striatal NPV → WW burst:      ρ={rho_npv_burst:.3f}, p={p_npv_burst:.4f}")
print(f"  ET fraction → WW burst:       ρ={rho_et_burst:.3f}, p={p_et_burst:.4f}")

print(f"\n{'='*60}")
print("L5-ONLY ANALYSIS (5 subregions, layer-controlled)")
print(f"{'='*60}")

l5_common = [r for r in common if r in L5_REGIONS]
npv_l5 = np.array([stri_npv[r] for r in l5_common])
et_l5 = np.array([et_fraction[r]["et_frac"] for r in l5_common])
rate_l5 = np.array([ww_rate[r] for r in l5_common])
burst_l5 = np.array([ww_burst[r] for r in l5_common])

if len(l5_common) >= 3:
    rho_npv_et_l5, p_npv_et_l5 = spearmanr(npv_l5, et_l5)
    rho_et_rate_l5, p_et_rate_l5 = spearmanr(et_l5, rate_l5)
    rho_npv_rate_l5, p_npv_rate_l5 = spearmanr(npv_l5, rate_l5)
    print(f"  Striatal NPV → ET fraction:   ρ={rho_npv_et_l5:.3f}, p={p_npv_et_l5:.4f}")
    print(f"  ET fraction → WW rate:        ρ={rho_et_rate_l5:.3f}, p={p_et_rate_l5:.4f}")
    print(f"  Striatal NPV → WW rate:       ρ={rho_npv_rate_l5:.3f}, p={p_npv_rate_l5:.4f}")
else:
    p_npv_et_l5 = 1.0
    p_npv_rate_l5 = 1.0
    rho_npv_et_l5 = 0
    rho_et_rate_l5 = 0
    rho_npv_rate_l5 = 0

print(f"\n{'='*60}")
print("L6-ONLY ANALYSIS (5 subregions, layer-controlled)")
print(f"{'='*60}")

l6_common = [r for r in common if r in L6_REGIONS]
npv_l6 = np.array([stri_npv[r] for r in l6_common])
et_l6 = np.array([et_fraction[r]["et_frac"] for r in l6_common])
rate_l6 = np.array([ww_rate[r] for r in l6_common])

if len(l6_common) >= 3:
    rho_npv_et_l6, p_npv_et_l6 = spearmanr(npv_l6, et_l6)
    rho_et_rate_l6, p_et_rate_l6 = spearmanr(et_l6, rate_l6)
    rho_npv_rate_l6, p_npv_rate_l6 = spearmanr(npv_l6, rate_l6)
    print(f"  Striatal NPV → ET fraction:   ρ={rho_npv_et_l6:.3f}, p={p_npv_et_l6:.4f}")
    print(f"  ET fraction → WW rate:        ρ={rho_et_rate_l6:.3f}, p={p_et_rate_l6:.4f}")
    print(f"  Striatal NPV → WW rate:       ρ={rho_npv_rate_l6:.3f}, p={p_npv_rate_l6:.4f}")
else:
    p_npv_et_l6 = 1.0
    p_npv_rate_l6 = 1.0

print(f"\n{'='*60}")
print("L5 vs L6 PAIRED CONTRAST (ET fraction & WW rate)")
print(f"{'='*60}")

l5_et_fracs = np.array([et_fraction[r]["et_frac"] for r in L5_REGIONS if r in et_fraction])
l6_et_fracs = np.array([et_fraction[r]["et_frac"] for r in L6_REGIONS if r in et_fraction])
l5_rates = np.array([ww_rate[r] for r in L5_REGIONS if r in ww_rate])
l6_rates = np.array([ww_rate[r] for r in L6_REGIONS if r in ww_rate])

if len(l5_et_fracs) >= 3 and len(l6_et_fracs) >= 3:
    stat_et, p_l5_gt_l6_et = wilcoxon(l5_et_fracs, l6_et_fracs, alternative="greater")
    stat_rate, p_l5_gt_l6_rate = wilcoxon(l5_rates, l6_rates, alternative="greater")
    print(f"  L5 > L6 ET fraction: Wilcoxon p={p_l5_gt_l6_et:.4f} (mean L5={l5_et_fracs.mean():.3f}, L6={l6_et_fracs.mean():.3f})")
    print(f"  L5 > L6 WW rate:     Wilcoxon p={p_l5_gt_l6_rate:.4f} (mean L5={l5_rates.mean():.3f}, L6={l6_rates.mean():.3f})")

n_perm = 10000
null_chain = []
for _ in range(n_perm):
    perm_et = np.random.permutation(et_vals)
    r1, _ = spearmanr(npv_vals, perm_et)
    r2, _ = spearmanr(perm_et, rate_vals)
    null_chain.append(min(r1, r2))
null_chain = np.array(null_chain)
obs_chain = min(rho_npv_et, rho_et_rate)
p_perm_chain = float(np.mean(null_chain >= obs_chain))
print(f"\n  Chain permutation (min link >= observed): p={p_perm_chain:.4f}")

chain_supported = (rho_npv_et > 0 and rho_et_rate > 0 and
                   (p_npv_et < 0.1 or p_et_rate < 0.1))
status = "CONFIRMED" if (p_perm_chain < 0.05 and chain_supported) else "REFUTED"

region_details = []
for r in common:
    layer = "L5" if r in L5_REGIONS else "L6"
    region_details.append({
        "region": r,
        "layer": layer,
        "striatal_npv": round(stri_npv[r], 4),
        "et_fraction": round(et_fraction[r]["et_frac"], 4),
        "it_fraction": round(et_fraction[r]["it_frac"], 4),
        "ct_fraction": round(et_fraction[r]["ct_frac"], 4),
        "ww_rate": round(ww_rate[r], 4),
        "ww_burst": round(ww_burst.get(r, 0), 4),
    })

results = {
    "hypothesis_id": "H21",
    "status": status,
    "full_npv_et_rho": float(rho_npv_et),
    "full_npv_et_p": float(p_npv_et),
    "full_et_rate_rho": float(rho_et_rate),
    "full_et_rate_p": float(p_et_rate),
    "full_npv_rate_rho": float(rho_npv_rate),
    "full_npv_rate_p": float(p_npv_rate),
    "full_npv_burst_rho": float(rho_npv_burst),
    "full_npv_burst_p": float(p_npv_burst),
    "l5_npv_et_rho": float(rho_npv_et_l5),
    "l5_npv_et_p": float(p_npv_et_l5),
    "l5_et_rate_rho": float(rho_et_rate_l5),
    "l5_et_rate_p": float(p_et_rate_l5),
    "l5_npv_rate_rho": float(rho_npv_rate_l5),
    "l5_npv_rate_p": float(p_npv_rate_l5),
    "l5_gt_l6_et_p": float(p_l5_gt_l6_et) if len(l5_et_fracs) >= 3 else None,
    "l5_gt_l6_rate_p": float(p_l5_gt_l6_rate) if len(l5_rates) >= 3 else None,
    "permutation_chain_p": float(p_perm_chain),
    "n_regions": len(common),
    "region_details": region_details,
    "notes": f"NPV→ET: ρ={rho_npv_et:.3f} p={p_npv_et:.3f}; ET→WWrate: ρ={rho_et_rate:.3f} p={p_et_rate:.3f}; chain_perm={p_perm_chain:.4f}; {status}"
}

print("\n" + json.dumps(results, indent=2))

out_path = OUTPUT_DIR / "results.json"
out_path.write_text(json.dumps(results, indent=2))

fig, axes = plt.subplots(2, 3, figsize=(18, 10))

def scatter_annot(ax, x, y, labels, xlabel, ylabel, title, rho, p):
    ax.scatter(x, y, s=100, zorder=5, edgecolors="black", linewidths=0.8)
    for i, lab in enumerate(labels):
        ax.annotate(lab, (x[i], y[i]), textcoords="offset points", xytext=(5, 3), fontsize=7)
    m, b = np.polyfit(x, y, 1)
    xl = np.linspace(x.min()*0.9, x.max()*1.1, 100)
    ax.plot(xl, m*xl+b, "--", color="gray", alpha=0.7)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(f"{title}\nρ={rho:.3f}, p={p:.3f}")

labels = [f"{'L5' if r in L5_REGIONS else 'L6'}:{r}" for r in common]

scatter_annot(axes[0,0], npv_vals, et_vals, labels,
    "Striatal NPV", "ET fraction", "Connectivity → ET Fraction", rho_npv_et, p_npv_et)
scatter_annot(axes[0,1], et_vals, rate_vals, labels,
    "ET fraction", "WW rate", "ET Fraction → WW Rate", rho_et_rate, p_et_rate)
scatter_annot(axes[0,2], npv_vals, rate_vals, labels,
    "Striatal NPV", "WW rate", "Connectivity → WW Rate", rho_npv_rate, p_npv_rate)

l5_labels = [r for r in l5_common]
l6_labels = [r for r in l6_common]

if len(l5_common) >= 3:
    scatter_annot(axes[1,0], npv_l5, et_l5, l5_labels,
        "Striatal NPV (L5 only)", "ET fraction", "L5: NPV → ET", rho_npv_et_l5, p_npv_et_l5)
    scatter_annot(axes[1,1], et_l5, rate_l5, l5_labels,
        "ET fraction (L5)", "WW rate", "L5: ET → WW Rate", rho_et_rate_l5, p_et_rate_l5)
    scatter_annot(axes[1,2], npv_l5, rate_l5, l5_labels,
        "Striatal NPV (L5)", "WW rate", "L5: NPV → WW Rate", rho_npv_rate_l5, p_npv_rate_l5)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "figures" / "striatal_et_ww.png", dpi=150)
plt.savefig(OUTPUT_DIR / "figures" / "striatal_et_ww.svg")

print(f"\nResults saved to {out_path}")

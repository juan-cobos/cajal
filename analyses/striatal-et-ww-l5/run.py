"""
Hypothesis H22: Within Layer 5 PFC, subregions with stronger striatal
projections have higher ET (extratelencephalic) Glut fractions and higher
WW firing rates. Layer is held constant (L5 only), removing the confound
from H21.

Three-way chain at single-layer resolution:
  striatal NPV → ET Glut fraction → WW firing rate

Branch: striatal-et-ww-l5
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

L5_REGIONS = ["PL5", "ILA5", "ORBl5", "ORBm5", "ORBvl5"]
L6_REGIONS = ["PL6a", "ILA6a", "ORBl6a", "ORBm6a", "ORBvl6a"]
PARENT_NAMES = ["PL", "ILA", "ORBl", "ORBm", "ORBvl"]

STRIATAL_ACRONYMS = ["STR", "STRd", "STRv", "CP", "ACB", "LS", "LSX", "BST", "SI"]
THALAMIC_IDS_MAP = {"MD": 362, "RE": 181, "PVT": 149, "VM": 685, "VL": 81, "AM": 127}


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
    for region in L5_REGIONS + L6_REGIONS:
        for f in sorted(cache.glob(f"{region}_*.parquet")):
            frames.append(pl.read_parquet(f).with_columns(pl.lit(region).alias("input_region")))
    return pl.concat(frames) if frames else pl.DataFrame()


def load_merfish_l5() -> pl.DataFrame:
    df = pl.read_csv(DATA_PATH / "merfish" / "cell_metadata_with_parcellation_annotation.csv")
    df = df.filter(pl.col("parcellation_substructure").is_in(L5_REGIONS))
    return df


def load_neural_activity_l5() -> pl.DataFrame:
    df = pl.read_csv(DATA_PATH / "pfcmap" / "units.csv")
    df = df.filter(pl.col("region").is_in(L5_REGIONS))
    df = df.filter(pl.col("layer") == "5")
    df = df.filter(pl.col("utype") == "ww")
    df = df.filter(pl.col("rate_mean").is_not_null())
    return df


print("=" * 60)
print("H22: L5-Only 3-Way Chain: Striatal NPV → ET → WW Rate")
print("=" * 60)

print("\n[1] Connectivity (L5 regions)...")
conn = load_connectivity_pfc()
stri_ids = load_striatal_ids()
conn_proj = conn.filter(pl.col("is_injection") == False)

stri_npv = {}
thal_npv = {}
for region in L5_REGIONS:
    sub = conn_proj.filter(pl.col("input_region") == region)
    stri_sub = sub.filter(pl.col("structure_id").is_in(list(stri_ids)))
    md_sub = sub.filter(pl.col("structure_id") == 362)
    stri_npv[region] = float(stri_sub["normalized_projection_volume"].mean()) if len(stri_sub) > 0 else 0
    thal_npv[region] = float(md_sub["normalized_projection_volume"].mean()) if len(md_sub) > 0 else 0
    print(f"  {region}: striatal NPV={stri_npv[region]:.4f}, MD NPV={thal_npv[region]:.4f}")

print("\n[2] MERFISH (L5 regions)...")
merfish = load_merfish_l5()
et_frac = {}
it_frac = {}
gaba_frac = {}
for region in L5_REGIONS:
    sub = merfish.filter(pl.col("parcellation_substructure") == region)
    glut = sub.filter(pl.col("neurotransmitter") == "Glut")
    sc = Counter(glut["subclass"].to_list())
    n_glut = len(glut)
    n_et = sum(v for k, v in sc.items() if "ET" in k)
    n_it = sum(v for k, v in sc.items() if "IT" in k)
    n_ct = sum(v for k, v in sc.items() if "CT" in k or "L6b" in k)
    n_np = sum(v for k, v in sc.items() if "NP" in k)
    et_frac[region] = n_et / n_glut if n_glut > 0 else 0
    it_frac[region] = n_it / n_glut if n_glut > 0 else 0
    nt_c = Counter(sub["neurotransmitter"].to_list())
    n_gaba = nt_c.get("GABA", 0)
    n_glut_nt = nt_c.get("Glut", 0)
    gaba_frac[region] = n_gaba / (n_gaba + n_glut_nt) if (n_gaba + n_glut_nt) > 0 else 0
    print(f"  {region}: ET={n_et}/{n_glut} ({et_frac[region]:.1%}), IT={it_frac[region]:.1%}, NP={n_np/n_glut:.1%}, GABA={gaba_frac[region]:.1%}")

print("\n[3] Neural Activity (L5, WW only)...")
neural = load_neural_activity_l5()
print(f"  Loaded {len(neural)} WW-L5 neurons")

ww_rate = {}
ww_burst = {}
for region in L5_REGIONS:
    sub = neural.filter(pl.col("region") == region)
    if len(sub) < 3:
        continue
    ww_rate[region] = float(sub["rate_mean"].mean())
    ww_burst[region] = float(sub["B_mean"].mean())
    print(f"  {region}: WW rate={ww_rate[region]:.3f}, burst={ww_burst[region]:.4f}, n={len(sub)}")

common = sorted(set(stri_npv.keys()) & set(et_frac.keys()) & set(ww_rate.keys()))
print(f"\n[4] L5 regions: {common}")

npv = np.array([stri_npv[r] for r in common])
et = np.array([et_frac[r] for r in common])
it = np.array([it_frac[r] for r in common])
gaba = np.array([gaba_frac[r] for r in common])
rate = np.array([ww_rate[r] for r in common])
burst = np.array([ww_burst[r] for r in common])
md = np.array([thal_npv[r] for r in common])

print(f"\n{'='*60}")
print("L5-ONLY CORRELATIONS (n=5)")
print(f"{'='*60}")

pairs = [
    ("Striatal NPV → ET frac", npv, et),
    ("Striatal NPV → IT frac", npv, it),
    ("Striatal NPV → GABA frac", npv, gaba),
    ("Striatal NPV → WW rate", npv, rate),
    ("Striatal NPV → WW burst", npv, burst),
    ("ET frac → WW rate", et, rate),
    ("ET frac → WW burst", et, burst),
    ("IT frac → WW rate", it, rate),
    ("IT frac → WW burst", it, burst),
    ("GABA frac → WW rate", gaba, rate),
    ("GABA frac → WW burst", gaba, burst),
    ("MD NPV → WW rate", md, rate),
    ("MD NPV → ET frac", md, et),
    ("MD NPV → GABA frac", md, gaba),
]

all_results = {}
for name, x, y in pairs:
    rho, p = spearmanr(x, y)
    pr = float(np.corrcoef(x, y)[0, 1])
    print(f"  {name:30s}: ρ={rho:+.3f}, p={p:.3f}, Pearson r={pr:+.3f}")
    all_results[name] = {"rho": float(rho), "p": float(p), "pearson_r": float(pr)}

n_perm = 10000
null_chain = []
for _ in range(n_perm):
    perm_et = np.random.permutation(et)
    r1, _ = spearmanr(npv, perm_et)
    r2, _ = spearmanr(perm_et, rate)
    null_chain.append(min(r1, r2))
null_chain = np.array(null_chain)
obs_chain_min = min(all_results["Striatal NPV → ET frac"]["rho"],
                    all_results["ET frac → WW rate"]["rho"])
p_perm_chain = float(np.mean(null_chain >= obs_chain_min))
print(f"\n  Chain permutation (min link): p={p_perm_chain:.4f}")

chain_l5 = (all_results["Striatal NPV → ET frac"]["rho"] > 0 and
            all_results["ET frac → WW rate"]["rho"] > 0 and
            (all_results["Striatal NPV → ET frac"]["p"] < 0.1 or
             all_results["ET frac → WW rate"]["p"] < 0.1))

status = "CONFIRMED" if (p_perm_chain < 0.05 and chain_l5) else "REFUTED"

print(f"\n[5] L6-ONLY COMPARISON")
l6_neural = pl.read_csv(DATA_PATH / "pfcmap" / "units.csv")
l6_neural = l6_neural.filter(pl.col("region").is_in(L6_REGIONS) & (pl.col("layer") == "6a") & (pl.col("utype") == "ww") & pl.col("rate_mean").is_not_null())
l6_rate = {}
for r in L6_REGIONS:
    sub = l6_neural.filter(pl.col("region") == r)
    if len(sub) >= 3:
        l6_rate[r] = float(sub["rate_mean"].mean())
        print(f"  {r}: WW rate={l6_rate[r]:.3f}, n={len(sub)}")

l6_stri = {}
for r in L6_REGIONS:
    sub = conn_proj.filter(pl.col("input_region") == r)
    stri_sub = sub.filter(pl.col("structure_id").is_in(list(stri_ids)))
    l6_stri[r] = float(stri_sub["normalized_projection_volume"].mean()) if len(stri_sub) > 0 else 0

l6_common = sorted(set(l6_stri.keys()) & set(l6_rate.keys()))
if len(l6_common) >= 3:
    l6_npv = np.array([l6_stri[r] for r in l6_common])
    l6_rate_arr = np.array([l6_rate[r] for r in l6_common])
    rho_l6, p_l6 = spearmanr(l6_npv, l6_rate_arr)
    print(f"  L6: Striatal NPV → WW rate: ρ={rho_l6:+.3f}, p={p_l6:.3f}")

region_details = []
for r in common:
    region_details.append({
        "region": r,
        "striatal_npv": round(stri_npv[r], 4),
        "md_npv": round(thal_npv[r], 4),
        "et_fraction": round(et_frac[r], 4),
        "it_fraction": round(it_frac[r], 4),
        "gaba_fraction": round(gaba_frac[r], 4),
        "ww_rate": round(ww_rate[r], 4),
        "ww_burst": round(ww_burst[r], 4),
    })

results = {
    "hypothesis_id": "H22",
    "status": status,
    "correlations": {k: v for k, v in all_results.items()},
    "permutation_chain_p": float(p_perm_chain),
    "n_regions": len(common),
    "layer": "5 only",
    "utype": "ww only",
    "region_details": region_details,
    "notes": f"L5-only: NPV→ET ρ={all_results['Striatal NPV → ET frac']['rho']:+.3f} p={all_results['Striatal NPV → ET frac']['p']:.3f}; ET→rate ρ={all_results['ET frac → WW rate']['rho']:+.3f} p={all_results['ET frac → WW rate']['p']:.3f}; chain_perm={p_perm_chain:.4f}; {status}"
}

print("\n" + json.dumps(results, indent=2))

out_path = OUTPUT_DIR / "results.json"
out_path.write_text(json.dumps(results, indent=2))

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

def scatter_annot(ax, x, y, labels, xlabel, ylabel, title, rho, p):
    ax.scatter(x, y, s=120, zorder=5, edgecolors="black", linewidths=0.8)
    for i, lab in enumerate(labels):
        ax.annotate(lab, (x[i], y[i]), textcoords="offset points", xytext=(6, 4), fontsize=9)
    m, b = np.polyfit(x, y, 1)
    xl = np.linspace(x.min()*0.9, x.max()*1.1, 100)
    ax.plot(xl, m*xl+b, "--", color="gray", alpha=0.7)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(f"{title}\nρ = {rho:.3f}, p = {p:.3f}")

labels = PARENT_NAMES[:len(common)]

r1 = all_results["Striatal NPV → ET frac"]
scatter_annot(axes[0,0], npv, et, labels, "Striatal NPV", "ET fraction", "L5: Striatal → ET", r1["rho"], r1["p"])
r2 = all_results["ET frac → WW rate"]
scatter_annot(axes[0,1], et, rate, labels, "ET fraction", "WW rate", "L5: ET → WW Rate", r2["rho"], r2["p"])
r3 = all_results["Striatal NPV → WW rate"]
scatter_annot(axes[1,0], npv, rate, labels, "Striatal NPV", "WW rate", "L5: Striatal → WW Rate", r3["rho"], r3["p"])
r4 = all_results["GABA frac → WW rate"]
scatter_annot(axes[1,1], gaba, rate, labels, "GABA fraction", "WW rate", "L5: GABA → WW Rate", r4["rho"], r4["p"])

plt.suptitle("H22: Layer-5 Only, WW Neurons — 3-Way Chain", fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "figures" / "striatal_et_ww_l5.png", dpi=150, bbox_inches="tight")
plt.savefig(OUTPUT_DIR / "figures" / "striatal_et_ww_l5.svg", bbox_inches="tight")

print(f"\nResults saved to {out_path}")

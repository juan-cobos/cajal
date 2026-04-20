"""
Hypothesis H08: PFC scRNA clusters show significant sex-biased cell-type
proportions beyond what is expected from the global male excess (67:33 M:F),
indicating preferential representation of specific transcriptomic types in
one sex within the prefrontal cortex.

Branch: scrna-sex-bias
Datasets: scrna
"""

import json
import random
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from scipy.stats import binomtest

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

DATA_PATH = Path(__file__).parent.parent.parent / "data"
OUTPUT_DIR = Path(__file__).parent


def load_scrna_pfc() -> pl.DataFrame:
    df = pl.read_csv(DATA_PATH / "scrna" / "cell_metadata.csv")
    df = df.filter(pl.col("region_of_interest_acronym") == "PL-ILA-ORB")
    return df


print("Loading scRNA PFC data...")
df = load_scrna_pfc()
print(f"Loaded {len(df)} PFC cells")

male = df.filter(pl.col("donor_sex") == "M")
female = df.filter(pl.col("donor_sex") == "F")
n_male = len(male)
n_female = len(female)
n_total = n_male + n_female
p_male_global = n_male / n_total

print(f"Male: {n_male} cells ({p_male_global:.1%})")
print(f"Female: {n_female} cells ({1 - p_male_global:.1%})")

cluster_counts = Counter(df["cluster_alias"].to_list())
all_clusters = sorted(cluster_counts.keys())
n_clusters = len(all_clusters)
print(f"Total clusters: {n_clusters}")

male_cluster_counts = Counter(male["cluster_alias"].to_list())
female_cluster_counts = Counter(female["cluster_alias"].to_list())

print("\n=== Per-cluster sex bias analysis ===")
cluster_results = []
for c in all_clusters:
    n_m = male_cluster_counts.get(c, 0)
    n_f = female_cluster_counts.get(c, 0)
    n_c = n_m + n_f
    if n_c < 10:
        continue
    p_val = binomtest(n_m, n_c, p_male_global).pvalue
    obs_male_frac = n_m / n_c
    bias_direction = "M" if obs_male_frac > p_male_global else "F"
    cluster_results.append({
        "cluster_alias": int(c),
        "n_male": n_m,
        "n_female": n_f,
        "n_total": n_c,
        "observed_male_frac": obs_male_frac,
        "expected_male_frac": p_male_global,
        "p_value": p_val,
        "bias_direction": bias_direction,
    })

cluster_results.sort(key=lambda x: x["p_value"])
n_tested = len(cluster_results)
print(f"Clusters tested (n >= 10): {n_tested}")

alpha = 0.05
p_values = np.array([r["p_value"] for r in cluster_results])

from statsmodels.stats.multitest import multipletests
reject, p_corrected, _, _ = multipletests(p_values, alpha=alpha, method="fdr_bh")

n_sig = int(np.sum(reject))
n_male_biased = sum(1 for i, r in enumerate(cluster_results) if reject[i] and r["bias_direction"] == "M")
n_female_biased = sum(1 for i, r in enumerate(cluster_results) if reject[i] and r["bias_direction"] == "F")

print(f"\nSignificant sex-biased clusters (FDR < {alpha}): {n_sig}/{n_tested}")
print(f"  Male-biased: {n_male_biased}")
print(f"  Female-biased: {n_female_biased}")

print(f"\nTop 15 sex-biased clusters:")
for i in range(min(15, n_tested)):
    r = cluster_results[i]
    sig = "*" if reject[i] else ""
    print(
        f"  alias={r['cluster_alias']}: M={r['n_male']}, F={r['n_female']}, "
        f"frac_M={r['observed_male_frac']:.3f}, p={r['p_value']:.2e}, "
        f"p_corr={p_corrected[i]:.2e} {r['bias_direction']}-bias{sig}"
    )

print("\n=== Permutation test for excess sex-biased clusters ===")
n_perm = 1000
null_n_sig = []
sex_arr = df["donor_sex"].to_list()
cluster_arr = df["cluster_alias"].to_list()

for perm_i in range(n_perm):
    sex_perm = np.random.permutation(sex_arr)
    m_perm = Counter()
    f_perm = Counter()
    for s, c in zip(sex_perm, cluster_arr):
        if s == "M":
            m_perm[c] += 1
        else:
            f_perm[c] += 1

    perm_pvals = []
    for c in all_clusters:
        n_m = m_perm.get(c, 0)
        n_f = f_perm.get(c, 0)
        n_c = n_m + n_f
        if n_c < 10:
            continue
        pv = binomtest(n_m, n_c, p_male_global).pvalue
        perm_pvals.append(pv)

    if len(perm_pvals) > 0:
        perm_pvals = np.array(perm_pvals)
        _, perm_corr, _, _ = multipletests(perm_pvals, alpha=alpha, method="fdr_bh")
        null_n_sig.append(int(np.sum(perm_corr < alpha)))
    else:
        null_n_sig.append(0)

    if (perm_i + 1) % 100 == 0:
        print(f"  Permutation {perm_i + 1}/{n_perm}")

null_n_sig = np.array(null_n_sig)
p_perm = float(np.mean(null_n_sig >= n_sig))
print(f"\nObserved significant clusters: {n_sig}")
print(f"Null distribution: mean={null_n_sig.mean():.1f}, 95th pct={np.percentile(null_n_sig, 95):.0f}")
print(f"Permutation p (null >= observed): {p_perm:.4f}")

status = "CONFIRMED" if p_perm < 0.05 else "REFUTED"

results = {
    "hypothesis_id": "H08",
    "status": status,
    "n_cells": n_total,
    "n_male": n_male,
    "n_female": n_female,
    "n_clusters_tested": n_tested,
    "n_sex_biased_clusters": n_sig,
    "n_male_biased": n_male_biased,
    "n_female_biased": n_female_biased,
    "permutation_p": p_perm,
    "null_mean_n_sig": float(null_n_sig.mean()),
    "null_95th_pct": float(np.percentile(null_n_sig, 95)),
    "notes": f"{n_sig} sex-biased clusters vs null mean={null_n_sig.mean():.1f}; p_perm={p_perm:.4f}; {status}"
}

print("\n" + json.dumps(results, indent=2))

out_path = OUTPUT_DIR / "results.json"
out_path.write_text(json.dumps(results, indent=2))

sizes = np.array([r["n_total"] for r in cluster_results])
male_fracs = np.array([r["observed_male_frac"] for r in cluster_results])
deviations = male_fracs - p_male_global

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].scatter(sizes, deviations, s=15, alpha=0.5, color="gray")
sig_mask = reject
axes[0].scatter(sizes[sig_mask], deviations[sig_mask], s=30, alpha=0.8, color="red", label="FDR significant", zorder=5)
axes[0].axhline(0, color="black", linestyle="--", alpha=0.5)
axes[0].set_xlabel("Cluster size (n cells)")
axes[0].set_ylabel("Male fraction deviation from global")
axes[0].set_title("Sex Bias vs Cluster Size")
axes[0].legend()

axes[1].hist(p_corrected, bins=50, color="steelblue", edgecolor="black", alpha=0.7)
axes[1].axvline(alpha, color="red", linestyle="--", label=f"FDR = {alpha}")
axes[1].set_xlabel("FDR-corrected p-value")
axes[1].set_ylabel("Count")
axes[1].set_title("Distribution of Corrected P-values")
axes[1].legend()

axes[2].hist(null_n_sig, bins=30, color="lightgray", edgecolor="black", alpha=0.7, label="Null")
axes[2].axvline(n_sig, color="red", linestyle="--", linewidth=2, label=f"Observed = {n_sig}")
axes[2].set_xlabel("Number of sex-biased clusters")
axes[2].set_ylabel("Count")
axes[2].set_title("Permutation Null Distribution")
axes[2].legend()

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "figures" / "scrna_sex_bias.png", dpi=150)
plt.savefig(OUTPUT_DIR / "figures" / "scrna_sex_bias.svg")

print(f"\nResults saved to {out_path}")
print("Figures saved to figures/")

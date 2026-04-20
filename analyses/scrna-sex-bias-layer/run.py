"""
Hypothesis H20: scRNA sex-biased clusters in PFC (from H08) are enriched
for specific neurotransmitter types (e.g., male-biased clusters are
predominantly glutamatergic, female-biased are GABAergic), reflecting
sex differences in cortical excitation/inhibition balance.

Branch: scrna-sex-bias-layer
Datasets: scrna, merfish
"""

import json
import random
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from scipy.stats import binomtest, fisher_exact

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

DATA_PATH = Path(__file__).parent.parent.parent / "data"
OUTPUT_DIR = Path(__file__).parent


def load_scrna_pfc() -> pl.DataFrame:
    df = pl.read_csv(DATA_PATH / "scrna" / "cell_metadata.csv")
    df = df.filter(pl.col("region_of_interest_acronym") == "PL-ILA-ORB")
    return df


print("=" * 60)
print("H20: scRNA Sex-Biased Clusters by Neurotransmitter Type")
print("=" * 60)

print("\nLoading scRNA data...")
df = load_scrna_pfc()
print(f"Loaded {len(df)} cells")

male = df.filter(pl.col("donor_sex") == "M")
female = df.filter(pl.col("donor_sex") == "F")
p_male_global = len(male) / len(df)
print(f"Global male fraction: {p_male_global:.3f}")

cluster_counts = Counter(df["cluster_alias"].to_list())
male_cluster_counts = Counter(male["cluster_alias"].to_list())
female_cluster_counts = Counter(female["cluster_alias"].to_list())

print("\nIdentifying sex-biased clusters...")
sex_biased = []
for c in sorted(cluster_counts.keys()):
    n_m = male_cluster_counts.get(c, 0)
    n_f = female_cluster_counts.get(c, 0)
    n_c = n_m + n_f
    if n_c < 10:
        continue
    p_val = binomtest(n_m, n_c, p_male_global).pvalue
    obs_frac = n_m / n_c
    direction = "M" if obs_frac > p_male_global else "F"
    sex_biased.append({
        "cluster_alias": int(c),
        "n_male": n_m,
        "n_female": n_f,
        "n_total": n_c,
        "male_frac": obs_frac,
        "p_value": p_val,
        "direction": direction,
    })

from statsmodels.stats.multitest import multipletests
p_values = np.array([r["p_value"] for r in sex_biased])
reject, p_corrected, _, _ = multipletests(p_values, alpha=0.05, method="fdr_bh")

for i, r in enumerate(sex_biased):
    r["fdr_significant"] = bool(reject[i])
    r["p_corrected"] = float(p_corrected[i])

sig_clusters = [r for r in sex_biased if r["fdr_significant"]]
m_biased = [r for r in sig_clusters if r["direction"] == "M"]
f_biased = [r for r in sig_clusters if r["direction"] == "F"]
print(f"Total sex-biased clusters: {len(sig_clusters)} (M-biased: {len(m_biased)}, F-biased: {len(f_biased)})")

print("\nLoading MERFISH data to get neurotransmitter labels...")
merfish = pl.read_csv(DATA_PATH / "merfish" / "cell_metadata_with_parcellation_annotation.csv")
pfc_parents = ["PL", "ILA", "ORBl", "ORBm", "ORBvl"]
pfc_pat = "|".join(pfc_parents)
merfish = merfish.filter(pl.col("parcellation_structure").str.contains(pfc_pat))

cluster_nt_map = {}
for row in merfish.iter_rows(named=True):
    alias = row.get("cluster_alias")
    nt = row.get("neurotransmitter")
    if alias is not None and nt is not None:
        if alias not in cluster_nt_map:
            cluster_nt_map[alias] = Counter()
        cluster_nt_map[alias][nt] += 1

cluster_nt_label = {}
for alias, counts in cluster_nt_map.items():
    dominant = counts.most_common(1)[0][0]
    cluster_nt_label[alias] = dominant

m_gaba = sum(1 for r in m_biased if cluster_nt_label.get(r["cluster_alias"]) == "GABA")
m_glut = sum(1 for r in m_biased if cluster_nt_label.get(r["cluster_alias"]) == "Glut")
m_other = len(m_biased) - m_gaba - m_glut
f_gaba = sum(1 for r in f_biased if cluster_nt_label.get(r["cluster_alias"]) == "GABA")
f_glut = sum(1 for r in f_biased if cluster_nt_label.get(r["cluster_alias"]) == "Glut")
f_other = len(f_biased) - f_gaba - f_glut

print(f"\nM-biased clusters: GABA={m_gaba}, Glut={m_glut}, Other={m_other}")
print(f"F-biased clusters: GABA={f_gaba}, Glut={f_glut}, Other={f_other}")

table = np.array([[m_gaba, m_glut], [f_gaba, f_glut]])
if table.sum() > 0 and min(table.shape) > 0:
    odd_ratio, p_fisher = fisher_exact(table, alternative="two-sided")
    print(f"Fisher's exact test: OR={odd_ratio:.3f}, p={p_fisher:.4f}")
else:
    odd_ratio, p_fisher = None, 1.0

m_gaba_frac = m_gaba / (m_gaba + m_glut) if (m_gaba + m_glut) > 0 else 0
f_gaba_frac = f_gaba / (f_gaba + f_glut) if (f_gaba + f_glut) > 0 else 0
print(f"M-biased GABA fraction: {m_gaba_frac:.3f}")
print(f"F-biased GABA fraction: {f_gaba_frac:.3f}")

hypothesis_prediction = m_gaba_frac < f_gaba_frac
status = "CONFIRMED" if (p_fisher < 0.05 and hypothesis_prediction) else "REFUTED"

results = {
    "hypothesis_id": "H20",
    "status": status,
    "m_biased_gaba": m_gaba,
    "m_biased_glut": m_glut,
    "f_biased_gaba": f_gaba,
    "f_biased_glut": f_glut,
    "m_biased_gaba_frac": float(m_gaba_frac),
    "f_biased_gaba_frac": float(f_gaba_frac),
    "fisher_p": float(p_fisher),
    "fisher_OR": float(odd_ratio) if odd_ratio else None,
    "n_sex_biased": len(sig_clusters),
    "n_m_biased": len(m_biased),
    "n_f_biased": len(f_biased),
    "n_clusters_mapped": len(cluster_nt_label),
    "notes": f"M-biased GABA frac={m_gaba_frac:.3f}, F-biased GABA frac={f_gaba_frac:.3f}; Fisher p={p_fisher:.4f}; {status}"
}

print("\n" + json.dumps(results, indent=2))

out_path = OUTPUT_DIR / "results.json"
out_path.write_text(json.dumps(results, indent=2))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

categories = ["GABA", "Glut", "Other"]
m_counts = [m_gaba, m_glut, m_other]
f_counts = [f_gaba, f_glut, f_other]

x_pos = np.arange(len(categories))
width = 0.35
axes[0].bar(x_pos - width/2, m_counts, width, label="M-biased", color="#4C72B0", edgecolor="black")
axes[0].bar(x_pos + width/2, f_counts, width, label="F-biased", color="#C44E52", edgecolor="black")
axes[0].set_xticks(x_pos)
axes[0].set_xticklabels(categories)
axes[0].set_ylabel("Number of clusters")
axes[0].set_title("Sex-Biased Clusters by NT Type")
axes[0].legend()

gaba_fracs = [m_gaba_frac, f_gaba_frac]
axes[1].bar(["M-biased", "F-biased"], gaba_fracs, color=["#4C72B0", "#C44E52"], edgecolor="black")
axes[1].set_ylabel("GABA fraction among NT-typed clusters")
axes[1].set_title(f"GABA Fraction in Sex-Biased Clusters\nFisher p = {p_fisher:.4f}")

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "figures" / "scrna_sex_bias_nt.png", dpi=150)
plt.savefig(OUTPUT_DIR / "figures" / "scrna_sex_bias_nt.png")

print(f"\nResults saved to {out_path}")

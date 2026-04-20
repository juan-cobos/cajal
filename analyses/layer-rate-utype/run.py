"""
Hypothesis H03: The firing rate difference between narrow-width and wide-width
PFC neurons is greater in deep layers (5/6) than in superficial layers (2/3).
Branch: layer-rate-utype
Datasets: neural_activity
"""

import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from scipy.stats import mannwhitneyu

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

DATA_PATH = Path(__file__).parent.parent.parent / "data"
OUTPUT_DIR = Path(__file__).parent


def load_neural_activity() -> pl.DataFrame:
    df = pl.read_csv(DATA_PATH / "pfcmap" / "units.csv")
    pfc_areas = ["ORBvl", "ORBl", "ORBm", "ILA", "PL"]
    df = df.filter(
        pl.col("region").str.contains("|".join(pfc_areas))
    )
    df = df.filter(pl.col("utype").is_in(["ww", "nw"]))
    df = df.filter(pl.col("rate_mean").is_not_null())
    df = df.filter(pl.col("layer").is_in(["2/3", "5", "6a", "6b"]))
    df = df.with_columns(
        pl.when(pl.col("layer") == "2/3")
        .then(pl.lit("superficial"))
        .otherwise(pl.lit("deep"))
        .alias("layer_group")
    )
    return df


def bootstrap_ci(data: np.ndarray, n_boot: int = 10000, ci: float = 0.95) -> tuple:
    boots = []
    for _ in range(n_boot):
        sample = np.random.choice(data, size=len(data), replace=True)
        boots.append(np.mean(sample))
    lo = np.percentile(boots, (1 - ci) / 2 * 100)
    hi = np.percentile(boots, (1 + ci) / 2 * 100)
    return float(np.mean(boots)), float(lo), float(hi)


def permutation_interaction_test(
    sup_ww: np.ndarray,
    sup_nw: np.ndarray,
    deep_ww: np.ndarray,
    deep_nw: np.ndarray,
    n_perm: int = 10000,
) -> tuple:
    observed_sup_diff = np.mean(sup_nw) - np.mean(sup_ww)
    observed_deep_diff = np.mean(deep_nw) - np.mean(deep_ww)
    observed_interaction = observed_deep_diff - observed_sup_diff

    all_rates = np.concatenate([sup_ww, sup_nw, deep_ww, deep_nw])
    n_sup_ww = len(sup_ww)
    n_sup_nw = len(sup_nw)
    n_deep_ww = len(deep_ww)
    n_deep_nw = len(deep_nw)

    null_interactions = []
    for _ in range(n_perm):
        perm = np.random.permutation(all_rates)
        p_sup_ww = perm[:n_sup_ww]
        p_sup_nw = perm[n_sup_ww : n_sup_ww + n_sup_nw]
        p_deep_ww = perm[n_sup_ww + n_sup_nw : n_sup_ww + n_sup_nw + n_deep_ww]
        p_deep_nw = perm[n_sup_ww + n_sup_nw + n_deep_ww :]

        perm_sup_diff = np.mean(p_sup_nw) - np.mean(p_sup_ww)
        perm_deep_diff = np.mean(p_deep_nw) - np.mean(p_deep_ww)
        null_interactions.append(perm_deep_diff - perm_sup_diff)

    null_interactions = np.array(null_interactions)
    p_value = float(np.mean(null_interactions >= observed_interaction))

    return observed_interaction, p_value


print("Loading neural activity data...")
df = load_neural_activity()
print(f"Loaded {len(df)} neurons")

sup = df.filter(pl.col("layer_group") == "superficial")
deep = df.filter(pl.col("layer_group") == "deep")

sup_ww = sup.filter(pl.col("utype") == "ww")["rate_mean"].to_numpy()
sup_nw = sup.filter(pl.col("utype") == "nw")["rate_mean"].to_numpy()
deep_ww = deep.filter(pl.col("utype") == "ww")["rate_mean"].to_numpy()
deep_nw = deep.filter(pl.col("utype") == "nw")["rate_mean"].to_numpy()

print(f"\nSuperficial: n_ww={len(sup_ww)}, n_nw={len(sup_nw)}")
print(f"Deep: n_ww={len(deep_ww)}, n_nw={len(deep_nw)}")

sup_diff = np.mean(sup_nw) - np.mean(sup_ww)
deep_diff = np.mean(deep_nw) - np.mean(deep_ww)
interaction = deep_diff - sup_diff

print(f"\n=== Layer-specific firing rate differences ===")
print(f"Superficial nw-ww diff: {sup_diff:.4f} (nw={np.mean(sup_nw):.4f}, ww={np.mean(sup_ww):.4f})")
print(f"Deep nw-ww diff: {deep_diff:.4f} (nw={np.mean(deep_nw):.4f}, ww={np.mean(deep_ww):.4f})")
print(f"Interaction (deep - superficial): {interaction:.4f}")

u_sup, p_sup = mannwhitneyu(sup_nw, sup_ww, alternative="greater")
u_deep, p_deep = mannwhitneyu(deep_nw, deep_ww, alternative="greater")
print(f"\nSuperficial: Mann-Whitney U={u_sup:.0f}, p={p_sup:.2e}")
print(f"Deep: Mann-Whitney U={u_deep:.0f}, p={p_deep:.2e}")

obs_int, p_int = permutation_interaction_test(sup_ww, sup_nw, deep_ww, deep_nw, n_perm=10000)
print(f"\nPermutation interaction test: observed={obs_int:.4f}, p={p_int:.4f}")

ci_sup = bootstrap_ci(sup_nw - sup_nw.mean() + sup_ww.mean() - sup_ww.mean())
ci_deep = bootstrap_ci(deep_nw - deep_nw.mean() + deep_ww.mean() - deep_ww.mean())

status = "CONFIRMED" if p_int < 0.05 and interaction > 0 else "REFUTED"

results = {
    "hypothesis_id": "H03",
    "status": status,
    "statistic": float(obs_int),
    "p_value": float(p_int),
    "effect_size": float(interaction),
    "n_cells": int(len(df)),
    "superficial_diff": float(sup_diff),
    "deep_diff": float(deep_diff),
    "interaction": float(interaction),
    "notes": f"Deep nw-ww diff ({deep_diff:.3f}) > Superficial ({sup_diff:.3f}); interaction p={p_int:.4f}"
}

print("\n" + json.dumps(results, indent=2))

out_path = OUTPUT_DIR / "results.json"
out_path.write_text(json.dumps(results, indent=2))

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

labels = ["Superficial\n(Layer 2/3)", "Deep\n(Layer 5/6)"]
diffs = [sup_diff, deep_diff]
colors = ["#4C72B0", "#DD8452"]
axes[0].bar(labels, diffs, color=colors, edgecolor="black")
axes[0].set_ylabel("Firing rate difference (nw - ww)")
axes[0].set_title("Firing Rate Difference by Layer\n(narrow-width minus wide-width)")
axes[0].axhline(0, color="black", linestyle="--", alpha=0.5)

bp_data = [sup_ww, sup_nw, deep_ww, deep_nw]
bp_labels = ["Sup WW", "Sup NW", "Deep WW", "Deep NW"]
bp = axes[1].boxplot(bp_data, labels=bp_labels, patch_artist=True)
bp_colors = ["#4C72B0", "#4C72B0", "#DD8452", "#DD8452"]
bp_hatches = ["", "//", "", "//"]
for patch, color, hatch in zip(bp["boxes"], bp_colors, bp_hatches):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
    patch.set_hatch(hatch)
axes[1].set_ylabel("Firing rate (log10 Hz)")
axes[1].set_title("Firing Rate Distribution\nby Layer and Subtype")

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "figures" / "layer_rate_utype.png", dpi=150)
plt.savefig(OUTPUT_DIR / "figures" / "layer_rate_utype.svg")

print(f"\nResults saved to {out_path}")
print("Figures saved to figures/")

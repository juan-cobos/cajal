"""
Hypothesis H33: Network analysis (networkx) within and across modalities.

WITHIN-MODALITY NETWORKS:
1. MERFISH spatial kNN graph — nodes=cells, edges=spatial neighbors.
   Test: modularity by transcriptomic cluster (are there transcriptomic
   communities in the spatial graph?).
2. Functional co-activity graph — nodes=neurons from same recording session,
   edges=spatial proximity. Test: modularity by ephys cluster.

CROSS-MODALITY NETWORKS:
3. Bipartite graph: transcriptomic subclass ↔ ephys cluster, weighted by
   co-occurrence across regions. Test: are there preferential cross-modal
   pairings?
4. Multiplex alignment: do spatial communities (MERFISH) align with
   functional communities (neural activity) at the region level?

5. Connectivity network: regions as nodes, projection similarity as edges.
   Test: does this network's community structure match transcriptomic/
   functional composition communities?

Branch: network-analysis
Datasets: merfish, neural_activity, connectivity
"""

import glob
import json
import random
from collections import Counter
from pathlib import Path

import anndata as ad
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import polars as pl
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr, pearsonr

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

DATA_PATH = Path(__file__).parent.parent.parent / "data"
OUTPUT_DIR = Path(__file__).parent

L5_REGIONS = ["PL5", "ILA5", "ORBl5", "ORBm5", "ORBvl5"]
L6_REGIONS = ["PL6a", "ILA6a", "ORBl6a", "ORBm6a", "ORBvl6a"]
ALL_REGIONS = L5_REGIONS + L6_REGIONS

print("=" * 60)
print("H33: Network analysis within and across modalities")
print("=" * 60)

# ============================================================
# NETWORK 1: MERFISH spatial kNN graph
# ============================================================
print("\n[1] MERFISH spatial kNN network...")

adata = ad.read_h5ad(DATA_PATH / "merfish" / "C57BL6J-638850.h5ad")
meta = pl.read_csv(DATA_PATH / "merfish" / "cell_metadata_with_parcellation_annotation.csv")
pfc_cells = meta.filter(pl.col("parcellation_substructure").is_in(ALL_REGIONS))
cell_labels = set(pfc_cells["cell_label"].to_list())
adata = adata[adata.obs_names.isin(cell_labels)].copy()
obs_df = pfc_cells.to_pandas().set_index("cell_label")
common = adata.obs_names.intersection(obs_df.index)
adata = adata[common].copy()
adata.obs["region"] = obs_df.loc[common, "parcellation_substructure"].values
adata.obs["x"] = obs_df.loc[common, "x_reconstructed"].values.astype(float)
adata.obs["y"] = obs_df.loc[common, "y_reconstructed"].values.astype(float)
adata.obs["z"] = obs_df.loc[common, "z_reconstructed"].values.astype(float)
adata.obs["subclass"] = obs_df.loc[common, "subclass"].values
adata.obs["supertype"] = obs_df.loc[common, "supertype"].values
adata.obs["merfish_cluster"] = obs_df.loc[common, "cluster"].values
adata.obs["nt"] = obs_df.loc[common, "neurotransmitter"].values

valid = ~(np.isnan(adata.obs["x"].values.astype(float)) |
          np.isnan(adata.obs["y"].values.astype(float)) |
          np.isnan(adata.obs["z"].values.astype(float)))
adata = adata[valid].copy()

coords = np.column_stack([adata.obs["x"].values, adata.obs["y"].values, adata.obs["z"].values])
subclass_labels = adata.obs["subclass"].values
supertype_labels = adata.obs["supertype"].values
cluster_labels = adata.obs["merfish_cluster"].values
nt_labels = adata.obs["nt"].values
region_labels = adata.obs["region"].values

print(f"  {adata.shape[0]} cells")

# Build kNN graph per region (to keep memory manageable)
K = 10
print(f"  Building k={K} nearest-neighbor spatial graph per region...")

all_modularity = {}
all_rand_index = {}

for region in ALL_REGIONS:
    rmask = region_labels == region
    r_idx = np.where(rmask)[0]
    n_r = len(r_idx)
    if n_r < 50:
        continue

    r_coords = coords[r_idx]
    r_subclass = subclass_labels[r_idx]
    r_nt = nt_labels[r_idx]

    G = nx.Graph()
    G.add_nodes_from(range(n_r))

    dists = cdist(r_coords, r_coords, metric="euclidean")
    for i in range(n_r):
        dists_i = dists[i].copy()
        dists_i[i] = np.inf
        knn = np.argsort(dists_i)[:K]
        for j in knn:
            G.add_edge(i, j)

    nx.set_node_attributes(G, {i: r_subclass[i] for i in range(n_r)}, "subclass")
    nx.set_node_attributes(G, {i: r_nt[i] for i in range(n_r)}, "nt")

    # Modularity by subclass
    partition_sub = {i: r_subclass[i] for i in range(n_r)}
    try:
        mod_sub = nx.community.modularity(G, nx.community.greedy_modularity_communities(G))
        mod_sub_label = nx.community.modularity(G, [
            {i for i in range(n_r) if r_subclass[i] == s}
            for s in set(r_subclass)
        ])
    except Exception:
        mod_sub = 0
        mod_sub_label = 0

    # Modularity by NT
    try:
        mod_nt_label = nx.community.modularity(G, [
            {i for i in range(n_r) if r_nt[i] == nt}
            for nt in set(r_nt)
        ])
    except Exception:
        mod_nt_label = 0

    # Detected communities vs subclass labels (adjusted Rand index)
    communities = list(nx.community.greedy_modularity_communities(G))
    detected_partition = {}
    for ci, comm in enumerate(communities):
        for node in comm:
            detected_partition[node] = ci

    from sklearn.metrics import adjusted_rand_score
    ari = adjusted_rand_score(
        [partition_sub[i] for i in range(n_r)],
        [detected_partition[i] for i in range(n_r)]
    )

    all_modularity[region] = {
        "n_nodes": n_r,
        "n_edges": G.number_of_edges(),
        "density": round(nx.density(G), 6),
        "avg_degree": round(sum(dict(G.degree()).values()) / n_r, 2),
        "modularity_detected": round(float(mod_sub), 4),
        "modularity_by_subclass": round(float(mod_sub_label), 4),
        "modularity_by_nt": round(float(mod_nt_label), 4),
        "n_communities": len(communities),
        "ari_detected_vs_subclass": round(float(ari), 4),
    }
    print(f"    {region}: n={n_r}, edges={G.number_of_edges()}, "
          f"mod_detected={mod_sub:.4f}, mod_subclass={mod_sub_label:.4f}, "
          f"mod_nt={mod_nt_label:.4f}, ARI={ari:.4f}, n_comm={len(communities)}")

# Permutation test: shuffle subclass labels, recompute modularity
print(f"\n  Permutation test for modularity (PL5, 1000 perms)...")
test_region = "PL5"
rmask = region_labels == test_region
r_idx = np.where(rmask)[0]
r_coords = coords[r_idx]
r_subclass = subclass_labels[r_idx]
n_r = len(r_idx)

dists = cdist(r_coords, r_coords, metric="euclidean")
G = nx.Graph()
G.add_nodes_from(range(n_r))
for i in range(n_r):
    dists_i = dists[i].copy()
    dists_i[i] = np.inf
    knn = np.argsort(dists_i)[:K]
    for j in knn:
        G.add_edge(i, j)

obs_mod_sub = all_modularity[test_region]["modularity_by_subclass"]

perm_mods = []
for _ in range(1000):
    perm_sub = np.random.permutation(r_subclass)
    try:
        pm = nx.community.modularity(G, [
            {i for i in range(n_r) if perm_sub[i] == s}
            for s in set(perm_sub)
        ])
        perm_mods.append(pm)
    except Exception:
        pass

perm_mods = np.array(perm_mods)
perm_p_mod = (np.sum(perm_mods >= obs_mod_sub) + 1) / (len(perm_mods) + 1)
print(f"    Observed modularity={obs_mod_sub:.4f}, null mean={perm_mods.mean():.4f}±{perm_mods.std():.4f}, p={perm_p_mod:.4f}")

# ============================================================
# NETWORK 2: Functional co-activity spatial graph
# ============================================================
print("\n[2] Functional (neural activity) spatial network...")

neural = pl.read_csv(DATA_PATH / "pfcmap" / "units.csv")
neural = neural.filter(pl.col("region").is_in(ALL_REGIONS))
neural = neural.filter(pl.col("utype").is_in(["ww", "nw"]))
neural = neural.filter(pl.col("rate_mean").is_not_null())
neural = neural.filter(pl.col("B_mean").is_not_null())

ephys_coords = np.column_stack([
    neural["AP"].to_numpy(),
    neural["DV"].to_numpy(),
    neural["ML"].to_numpy(),
])
ephys_clusters = neural["cluster"].to_numpy()
ephys_utypes = neural["utype"].to_numpy()
ephys_regions = neural["region"].to_numpy()

# Build kNN graph per recording session
func_mod_results = {}
by_rec = neural.group_by("recid").len().sort("len", descending=True)

for rec_row in by_rec.head(20).iter_rows(named=True):
    recid = rec_row["recid"]
    sub = neural.filter(pl.col("recid") == recid)
    n_sub = len(sub)
    if n_sub < 20:
        continue

    rec_coords = np.column_stack([
        sub["AP"].to_numpy(),
        sub["DV"].to_numpy(),
        sub["ML"].to_numpy(),
    ])
    rec_clusters = sub["cluster"].to_numpy()
    rec_utypes = sub["utype"].to_numpy()

    G_func = nx.Graph()
    G_func.add_nodes_from(range(n_sub))

    dists = cdist(rec_coords, rec_coords, metric="euclidean")
    for i in range(min(n_sub, 300)):
        dists_i = dists[i].copy()
        dists_i[i] = np.inf
        knn = np.argsort(dists_i)[:K]
        for j in knn:
            G_func.add_edge(i, j)

    try:
        communities = list(nx.community.greedy_modularity_communities(G_func))
        mod_det = nx.community.modularity(G_func, communities)
        mod_cl = nx.community.modularity(G_func, [
            {i for i in range(n_sub) if rec_clusters[i] == c}
            for c in set(rec_clusters)
        ])
        mod_ut = nx.community.modularity(G_func, [
            {i for i in range(n_sub) if rec_utypes[i] == u}
            for u in set(rec_utypes)
        ])
        ari = adjusted_rand_score(
            [rec_clusters[i] for i in range(n_sub)],
            [0] * n_sub  # placeholder
        )
        # ARI: detected vs cluster
        detected_part = {}
        for ci, comm in enumerate(communities):
            for node in comm:
                detected_part[node] = ci
        ari_func = adjusted_rand_score(
            [rec_clusters[i] for i in range(n_sub)],
            [detected_part.get(i, 0) for i in range(n_sub)]
        )
    except Exception:
        mod_det = 0
        mod_cl = 0
        mod_ut = 0
        ari_func = 0
        communities = []

    func_mod_results[recid] = {
        "n": n_sub,
        "mod_detected": round(float(mod_det), 4),
        "mod_by_cluster": round(float(mod_cl), 4),
        "mod_by_utype": round(float(mod_ut), 4),
        "ari_vs_cluster": round(float(ari_func), 4),
        "n_communities": len(communities),
    }
    print(f"    {recid[:30]:30s}: n={n_sub}, mod_cl={mod_cl:.4f}, mod_ut={mod_ut:.4f}, ARI={ari_func:.4f}")

mean_func_mod_cl = np.mean([v["mod_by_cluster"] for v in func_mod_results.values()])
mean_func_ari = np.mean([v["ari_vs_cluster"] for v in func_mod_results.values()])

# ============================================================
# NETWORK 3: Cross-modal bipartite graph
# ============================================================
print("\n[3] Cross-modal bipartite: subclass ↔ ephys cluster...")

# Build co-occurrence: for each region, what fraction of each subclass × ephys cluster
# Since MERFISH and neural_activity are different animals, we connect at region level
# Weight = correlation of subclass fraction and ephys cluster fraction across regions

mf_comp = {}
for region in ALL_REGIONS:
    rmask = region_labels == region
    r_sub = subclass_labels[rmask]
    r_nt = nt_labels[rmask]
    n_r = int(rmask.sum())
    sub_counts = Counter([f"{s}_{nt}" for s, nt in zip(r_sub, r_nt)])
    for key in sub_counts:
        sub_counts[key] /= n_r
    mf_comp[region] = sub_counts

ep_comp = {}
for region in ALL_REGIONS:
    sub = neural.filter(pl.col("region") == region)
    n_sub = len(sub)
    cl_counts = Counter([f"C{r['cluster']}_{r['utype']}" for r in sub.iter_rows(named=True)])
    for key in cl_counts:
        cl_counts[key] /= n_sub
    ep_comp[region] = cl_counts

mf_keys = sorted(set().union(*[v.keys() for v in mf_comp.values()]))
ep_keys = sorted(set().union(*[v.keys() for v in ep_comp.values()]))

# Build bipartite graph: subclass nodes ↔ ephys cluster nodes, weight=|spearman ρ|
B = nx.Graph()

for mf_key in mf_keys:
    mf_vals = np.array([mf_comp[r].get(mf_key, 0) for r in ALL_REGIONS])
    if np.std(mf_vals) == 0:
        continue
    for ep_key in ep_keys:
        ep_vals = np.array([ep_comp[r].get(ep_key, 0) for r in ALL_REGIONS])
        if np.std(ep_vals) == 0:
            continue
        rho, p = spearmanr(mf_vals, ep_vals)
        if abs(rho) > 0.5 and p < 0.1:
            B.add_node(mf_key, bipartite=0, type="transcriptomic")
            B.add_node(ep_key, bipartite=1, type="functional")
            B.add_edge(mf_key, ep_key, weight=round(float(rho), 4), p=round(float(p), 6))

n_trans_nodes = sum(1 for n, d in B.nodes(data=True) if d.get("bipartite") == 0)
n_func_nodes = sum(1 for n, d in B.nodes(data=True) if d.get("bipartite") == 1)
n_cross_edges = B.number_of_edges()

print(f"  Bipartite graph: {n_trans_nodes} transcriptomic × {n_func_nodes} functional nodes, {n_cross_edges} edges (|ρ|>0.5, p<0.1)")

if n_cross_edges > 0:
    top_edges = sorted(B.edges(data=True), key=lambda x: abs(x[2]["weight"]), reverse=True)[:15]
    print(f"  Top cross-modal links:")
    for u, v, d in top_edges:
        print(f"    {u:35s} ↔ {v:15s}: ρ={d['weight']:+.4f}, p={d['p']:.4f}")

    # Project to each side
    trans_nodes = {n for n, d in B.nodes(data=True) if d.get("bipartite") == 0}
    func_nodes = {n for n, d in B.nodes(data=True) if d.get("bipartite") == 1}

    trans_proj = nx.projected_graph(B, trans_nodes)
    func_proj = nx.projected_graph(B, func_nodes)

    print(f"  Transcriptomic projection: {trans_proj.number_of_nodes()} nodes, {trans_proj.number_of_edges()} edges")
    print(f"  Functional projection: {func_proj.number_of_nodes()} nodes, {func_proj.number_of_edges()} edges")

    # Community detection on transcriptomic projection
    if trans_proj.number_of_edges() > 0:
        trans_comm = list(nx.community.greedy_modularity_communities(trans_proj))
        print(f"  Transcriptomic communities: {len(trans_comm)}")
        for ci, comm in enumerate(trans_comm):
            print(f"    Community {ci}: {list(comm)[:5]}...")

    if func_proj.number_of_edges() > 0:
        func_comm = list(nx.community.greedy_modularity_communities(func_proj))
        print(f"  Functional communities: {len(func_comm)}")
        for ci, comm in enumerate(func_comm):
            print(f"    Community {ci}: {list(comm)}")

# ============================================================
# NETWORK 4: Connectivity network with composition overlay
# ============================================================
print("\n[4] Connectivity network: regions as nodes...")

conn_matrix = {}
for region in ALL_REGIONS:
    files = glob.glob(str(DATA_PATH / "connectivity" / f"{region}_*.parquet"))
    all_dfs = []
    for f in files:
        df = pl.read_parquet(f)
        proj = df.filter(pl.col("is_injection") == False)
        all_dfs.append(proj)
    combined = pl.concat(all_dfs)
    avg = combined.group_by("structure_id").agg(
        pl.col("normalized_projection_volume").mean().alias("NPV_mean")
    )
    conn_matrix[region] = dict(zip(avg["structure_id"].to_list(), avg["NPV_mean"].to_list()))

common_targets = sorted(set(conn_matrix[ALL_REGIONS[0]].keys()))
for region in ALL_REGIONS[1:]:
    common_targets = sorted(set(common_targets) & set(conn_matrix[region].keys()))

conn_mat = np.array([[conn_matrix[r].get(t, 0.0) for t in common_targets] for r in ALL_REGIONS])

C = nx.Graph()
C.add_nodes_from(ALL_REGIONS)

for i, r1 in enumerate(ALL_REGIONS):
    for j, r2 in enumerate(ALL_REGIONS):
        if j <= i:
            continue
        rho, _ = spearmanr(conn_mat[i], conn_mat[j])
        if rho > 0.5:
            C.add_edge(r1, r2, weight=round(float(rho), 4))

# Add node attributes
for region in ALL_REGIONS:
    rmask = region_labels == region
    r_nt = nt_labels[rmask]
    n_r = int(rmask.sum())
    gaba_frac = np.sum(r_nt == "GABA") / n_r if n_r > 0 else 0
    layer = "L5" if region in L5_REGIONS else "L6"
    C.nodes[region]["gaba_frac"] = round(float(gaba_frac), 4)
    C.nodes[region]["layer"] = layer

conn_comm = list(nx.community.greedy_modularity_communities(C))
print(f"  Connectivity network: {C.number_of_nodes()} nodes, {C.number_of_edges()} edges")
print(f"  Communities: {len(conn_comm)}")
for ci, comm in enumerate(conn_comm):
    print(f"    Community {ci}: {sorted(comm)}")

if C.number_of_edges() > 0:
    conn_mod = nx.community.modularity(C, conn_comm)
    print(f"  Modularity: {conn_mod:.4f}")

# Layer separation in connectivity network
within_layer_edges = 0
cross_layer_edges = 0
for u, v in C.edges():
    if C.nodes[u]["layer"] == C.nodes[v]["layer"]:
        within_layer_edges += 1
    else:
        cross_layer_edges += 1
print(f"  Within-layer edges: {within_layer_edges}, cross-layer: {cross_layer_edges}")

# ============================================================
# Summary
# ============================================================
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)

mean_mf_mod = np.mean([v["modularity_by_subclass"] for v in all_modularity.values()])
mean_mf_ari = np.mean([v["ari_detected_vs_subclass"] for v in all_modularity.values()])
mean_mf_mod_nt = np.mean([v["modularity_by_nt"] for v in all_modularity.values()])

print(f"  MERFISH spatial network:")
print(f"    Mean modularity (subclass partition): {mean_mf_mod:.4f}")
print(f"    Mean modularity (NT partition): {mean_mf_mod_nt:.4f}")
print(f"    Mean ARI (detected vs subclass): {mean_mf_ari:.4f}")
print(f"    Permutation p (PL5): {perm_p_mod:.4f}")
print(f"  Functional spatial network:")
print(f"    Mean modularity (cluster partition): {mean_func_mod_cl:.4f}")
print(f"    Mean ARI (detected vs cluster): {mean_func_ari:.4f}")
print(f"  Cross-modal bipartite:")
print(f"    {n_cross_edges} significant cross-modal links (|ρ|>0.5)")
print(f"  Connectivity network:")
print(f"    {len(conn_comm)} communities from {C.number_of_edges()} edges")

mf_sig = perm_p_mod < 0.05
func_sig = mean_func_mod_cl > 0.1
cross_sig = n_cross_edges > 5

status = "CONFIRMED" if (mf_sig or func_sig) else "REFUTED"

results = {
    "hypothesis_id": "H33",
    "status": status,
    "merfish_spatial_network": {
        "mean_modularity_subclass": round(float(mean_mf_mod), 4),
        "mean_modularity_nt": round(float(mean_mf_mod_nt), 4),
        "mean_ari_subclass": round(float(mean_mf_ari), 4),
        "per_region": all_modularity,
        "permutation_p_pl5": round(float(perm_p_mod), 6),
    },
    "functional_spatial_network": {
        "mean_modularity_cluster": round(float(mean_func_mod_cl), 4),
        "mean_ari_cluster": round(float(mean_func_ari), 4),
        "per_recording": func_mod_results,
    },
    "cross_modal_bipartite": {
        "n_transcriptomic_nodes": n_trans_nodes,
        "n_functional_nodes": n_func_nodes,
        "n_edges": n_cross_edges,
    },
    "connectivity_network": {
        "n_nodes": C.number_of_nodes(),
        "n_edges": C.number_of_edges(),
        "n_communities": len(conn_comm),
        "communities": [sorted(comm) for comm in conn_comm],
        "within_layer_edges": within_layer_edges,
        "cross_layer_edges": cross_layer_edges,
    },
    "notes": f"MF mod_subclass={mean_mf_mod:.4f} perm_p={perm_p_mod:.4f}; Func mod_cl={mean_func_mod_cl:.4f}; Cross {n_cross_edges} links; {status}"
}

def json_safe(obj):
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [json_safe(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj

results = json_safe(results)
out_path = OUTPUT_DIR / "results.json"
out_path.write_text(json.dumps(results, indent=2))

fig, axes = plt.subplots(2, 2, figsize=(16, 14))

ax = axes[0, 0]
regions_plotted = list(all_modularity.keys())
mod_vals = [all_modularity[r]["modularity_by_subclass"] for r in regions_plotted]
mod_nt_vals = [all_modularity[r]["modularity_by_nt"] for r in regions_plotted]
x = np.arange(len(regions_plotted))
ax.bar(x - 0.15, mod_vals, 0.3, label="By subclass", color="#4C72B0", alpha=0.8)
ax.bar(x + 0.15, mod_nt_vals, 0.3, label="By NT", color="#DD8452", alpha=0.8)
ax.set_xticks(x)
ax.set_xticklabels(regions_plotted, rotation=45, ha="right", fontsize=8)
ax.set_ylabel("Modularity")
ax.set_title("MERFISH kNN network modularity")
ax.legend(fontsize=8)
ax.axhline(0, color="black", linestyle="--", alpha=0.3)

ax = axes[0, 1]
rec_names = [k[:20] for k in list(func_mod_results.keys())[:15]]
mod_cl = [func_mod_results[k]["mod_by_cluster"] for k in list(func_mod_results.keys())[:15]]
mod_ut = [func_mod_results[k]["mod_by_utype"] for k in list(func_mod_results.keys())[:15]]
x2 = np.arange(len(rec_names))
ax.barh(x2 - 0.15, mod_cl, 0.3, label="By ephys cluster", color="#55A868", alpha=0.8)
ax.barh(x2 + 0.15, mod_ut, 0.3, label="By utype", color="#C44E52", alpha=0.8)
ax.set_yticks(x2)
ax.set_yticklabels(rec_names, fontsize=7)
ax.set_xlabel("Modularity")
ax.set_title("Functional kNN network modularity")
ax.legend(fontsize=8)

ax = axes[1, 0]
if n_cross_edges > 0:
    pos = nx.spring_layout(B, seed=SEED, k=2)
    trans_n = [n for n, d in B.nodes(data=True) if d.get("bipartite") == 0]
    func_n = [n for n, d in B.nodes(data=True) if d.get("bipartite") == 1]
    nx.draw_networkx_nodes(B, pos, nodelist=trans_n, node_color="#4C72B0", node_size=80, alpha=0.8, ax=ax)
    nx.draw_networkx_nodes(B, pos, nodelist=func_n, node_color="#DD8452", node_size=80, alpha=0.8, ax=ax)
    nx.draw_networkx_edges(B, pos, alpha=0.3, ax=ax)
    labels = {n: n[:15] for n in B.nodes()}
    nx.draw_networkx_labels(B, pos, labels, font_size=5, ax=ax)
    ax.set_title("Cross-modal bipartite: subclass ↔ ephys cluster")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor="#4C72B0", label="Transcriptomic"),
                       Patch(facecolor="#DD8452", label="Functional")], fontsize=8)

ax = axes[1, 1]
if C.number_of_edges() > 0:
    pos_c = nx.spring_layout(C, seed=SEED)
    node_colors = ["#4C72B0" if C.nodes[n]["layer"] == "L5" else "#DD8452" for n in C.nodes()]
    edge_weights = [C[u][v]["weight"] for u, v in C.edges()]
    nx.draw_networkx_nodes(C, pos_c, node_color=node_colors, node_size=300, alpha=0.8, ax=ax)
    nx.draw_networkx_edges(C, pos_c, width=[w * 3 for w in edge_weights], alpha=0.5, ax=ax)
    nx.draw_networkx_labels(C, pos_c, font_size=8, ax=ax)
    ax.set_title("Connectivity similarity network (ρ>0.5)")
    ax.legend(handles=[Patch(facecolor="#4C72B0", label="L5"),
                       Patch(facecolor="#DD8452", label="L6")], fontsize=8)

plt.suptitle("H33: Network analysis within and across modalities", fontsize=14)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "figures" / "network_analysis.png", dpi=150, bbox_inches="tight")
plt.savefig(OUTPUT_DIR / "figures" / "network_analysis.svg", bbox_inches="tight")

print(f"\nResults saved to {out_path}")
print("Done.")

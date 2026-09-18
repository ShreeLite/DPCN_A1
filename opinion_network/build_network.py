"""
Opinion Network Construction Pipeline
======================================
Assignment 1: Opinion Network Formation

This script implements the "Pipeline Followed" component of the report:
loading and cleaning the survey, encoding Likert responses, computing
pairwise opinion similarity between respondents, and constructing both
an overall opinion network and four topic-specific sub-networks
(Technology / Education / Ethics-Society / Environment).

Author: Shreem — Network Construction workstream
"""

import pandas as pd
import numpy as np
import networkx as nx
import community as community_louvain   # python-louvain
import matplotlib.pyplot as plt
import json
import os

OUT_DIR = "outputs"
FIG_DIR = "figures"
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

RNG_SEED = 42
np.random.seed(RNG_SEED)

# ---------------------------------------------------------------------
# STEP 1 — LOAD & CLEAN
# ---------------------------------------------------------------------

RAW_PATH = "../Survey_Results_UC.csv"

df = pd.read_csv(RAW_PATH)
df = df.rename(columns={df.columns[0]: "respondent_id"})

question_cols = df.columns[1:]

# Map topic prefixes to human-readable topic names
TOPIC_MAP = {"T": "Technology", "E": "Education", "S": "Ethics_Society", "V": "Environment"}
col_topic = {c: TOPIC_MAP[c[0]] for c in question_cols}

# "No Comments" is a non-response, not a neutral opinion -> treat as missing
df[question_cols] = df[question_cols].replace("No Comments", np.nan)

# Drop respondents who left more than half the survey blank (junk/incomplete rows)
missing_frac = df[question_cols].isna().mean(axis=1)
n_before = len(df)
df = df[missing_frac <= 0.5].reset_index(drop=True)
n_after = len(df)
print(f"[clean] Dropped {n_before - n_after} respondents with >50% missing answers "
      f"({n_before} -> {n_after} respondents kept).")

# Encode 5-point Likert scale as ordinal numeric values (-2 .. +2)
LIKERT_MAP = {
    "Strongly Disagree": -2,
    "Disagree": -1,
    "Neutral": 0,
    "Agree": 1,
    "Strongly Agree": 2,
}
encoded = df[question_cols].apply(lambda col: col.map(LIKERT_MAP))

# Impute remaining (sparse, <=50% missing) gaps with the item's column mean
# (i.e. the "average class opinion" on that statement) — a conservative,
# neutral choice that does not manufacture a fake strong opinion.
encoded = encoded.apply(lambda col: col.fillna(col.mean()), axis=0)

encoded.insert(0, "respondent_id", df["respondent_id"].values)
encoded.to_csv(os.path.join(OUT_DIR, "encoded_responses.csv"), index=False)

print(f"[clean] Final matrix: {encoded.shape[0]} respondents x "
      f"{encoded.shape[1]-1} items.")

# ---------------------------------------------------------------------
# STEP 2 — PAIRWISE OPINION SIMILARITY
# ---------------------------------------------------------------------
# We use Pearson correlation between respondents' opinion profiles rather
# than raw cosine/euclidean distance. Rationale: correlation captures
# whether two people's opinions rise and fall *together* across statements
# (shared opinion "shape"), and is insensitive to a respondent's personal
# tendency to answer more/less extremely overall — which is a scale bias,
# not a real opinion difference.

def build_similarity(matrix_df, cols):
    """Returns an (n x n) Pearson correlation matrix of respondents over the
    given item columns.

    Edge case: a respondent who gave the *same* answer to every item in a
    block (e.g. "Agree" to all 15 Environment items) has zero variance,
    which makes Pearson correlation undefined (0/0 -> NaN). This is a
    genuine "flat opinion profile", not a data error, so rather than
    dropping the respondent we treat their similarity to everyone else on
    that topic as 0 (neither agreement nor disagreement can be inferred).
    """
    X = matrix_df[cols].values.astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        sim = np.corrcoef(X)
    sim = np.nan_to_num(sim, nan=0.0)
    np.fill_diagonal(sim, 1.0)
    return sim

overall_cols = list(question_cols)
sim_overall = build_similarity(encoded, overall_cols)

topic_cols = {t: [c for c in question_cols if col_topic[c] == t] for t in TOPIC_MAP.values()}
sim_topic = {t: build_similarity(encoded, cols) for t, cols in topic_cols.items()}

ids = encoded["respondent_id"].astype(str).values
n = len(ids)

# ---------------------------------------------------------------------
# STEP 3 — GRAPH CONSTRUCTION (mutual k-NN sparsification)
# ---------------------------------------------------------------------
# A fully-connected similarity graph (n*(n-1)/2 edges) is not analytically
# useful or visualizable. We sparsify using mutual k-Nearest-Neighbours:
# an edge (i, j) is kept only if j is among i's top-K most similar peers
# AND i is among j's top-K most similar peers. This produces a graph that
# reflects genuinely reciprocal opinion closeness rather than one-sided
# similarity, and keeps the network sparse enough to interpret visually.

K = 8  # each respondent's k most opinion-similar peers

def build_knn_graph(sim_matrix, ids, k=K, label="graph"):
    n = len(ids)
    knn_sets = []
    for i in range(n):
        row = sim_matrix[i].copy()
        row[i] = -np.inf
        top_k = np.argsort(row)[-k:]
        knn_sets.append(set(top_k))

    G = nx.Graph(name=label)
    G.add_nodes_from(ids)
    for i in range(n):
        for j in knn_sets[i]:
            if i in knn_sets[j] and i < j:  # mutual + avoid double-adding
                G.add_edge(ids[i], ids[j], weight=float(sim_matrix[i, j]))
    return G

G_overall = build_knn_graph(sim_overall, ids, k=K, label="Overall Opinion Network")

topic_graphs = {
    t: build_knn_graph(sim_topic[t], ids, k=K, label=f"{t} Opinion Network")
    for t in TOPIC_MAP.values()
}

print(f"[graph] Overall network: {G_overall.number_of_nodes()} nodes, "
      f"{G_overall.number_of_edges()} edges, "
      f"density={nx.density(G_overall):.3f}")
for t, g in topic_graphs.items():
    print(f"[graph] {t}: {g.number_of_nodes()} nodes, {g.number_of_edges()} edges, "
          f"density={nx.density(g):.3f}")

# ---------------------------------------------------------------------
# STEP 4 — METRICS
# ---------------------------------------------------------------------

def graph_metrics(G):
    metrics = {}
    metrics["n_nodes"] = G.number_of_nodes()
    metrics["n_edges"] = G.number_of_edges()
    metrics["density"] = nx.density(G)
    metrics["avg_clustering"] = nx.average_clustering(G, weight="weight")
    if nx.is_connected(G):
        metrics["avg_path_length"] = nx.average_shortest_path_length(G)
        metrics["diameter"] = nx.diameter(G)
    else:
        largest_cc = max(nx.connected_components(G), key=len)
        Gc = G.subgraph(largest_cc)
        metrics["avg_path_length_largest_cc"] = nx.average_shortest_path_length(Gc)
        metrics["diameter_largest_cc"] = nx.diameter(Gc)
        metrics["n_components"] = nx.number_connected_components(G)
        metrics["largest_cc_size"] = len(largest_cc)
    return metrics

overall_metrics = graph_metrics(G_overall)
topic_metrics = {t: graph_metrics(g) for t, g in topic_graphs.items()}

with open(os.path.join(OUT_DIR, "network_metrics.json"), "w") as f:
    json.dump({"overall": overall_metrics, "by_topic": topic_metrics}, f, indent=2)

print("[metrics] Overall network metrics:", overall_metrics)

# ---------------------------------------------------------------------
# STEP 5 — CENTRALITY & COMMUNITY DETECTION (overall network)
# ---------------------------------------------------------------------

degree_centrality = nx.degree_centrality(G_overall)
betweenness = nx.betweenness_centrality(G_overall, weight="weight")
eigenvector = nx.eigenvector_centrality(G_overall, weight="weight", max_iter=1000)

partition = community_louvain.best_partition(G_overall, weight="weight", random_state=RNG_SEED)
modularity = community_louvain.modularity(partition, G_overall, weight="weight")
n_communities = len(set(partition.values()))
print(f"[community] Louvain found {n_communities} communities, modularity={modularity:.3f}")

centrality_df = pd.DataFrame({
    "respondent_id": ids,
    "degree_centrality": [degree_centrality[i] for i in ids],
    "betweenness_centrality": [betweenness[i] for i in ids],
    "eigenvector_centrality": [eigenvector[i] for i in ids],
    "community": [partition[i] for i in ids],
})
centrality_df.to_csv(os.path.join(OUT_DIR, "node_centrality_communities.csv"), index=False)

# Also detect communities per topic sub-network, for cross-topic comparison
topic_partitions = {
    t: community_louvain.best_partition(g, weight="weight", random_state=RNG_SEED)
    for t, g in topic_graphs.items()
}

# Compare topic communities to the overall community structure using
# Adjusted Rand Index (do the same people cluster together on each topic
# as they do overall?)
from sklearn.metrics import adjusted_rand_score

ari_results = {}
overall_labels = [partition[i] for i in ids]
for t, tp in topic_partitions.items():
    topic_labels = [tp[i] for i in ids]
    ari_results[t] = adjusted_rand_score(overall_labels, topic_labels)

with open(os.path.join(OUT_DIR, "cross_topic_ari.json"), "w") as f:
    json.dump(ari_results, f, indent=2)

print("[ari] Similarity of each topic's community structure to the overall network:")
for t, v in ari_results.items():
    print(f"   {t}: ARI = {v:.3f}")

# ---------------------------------------------------------------------
# STEP 6 — VISUALIZATIONS
# ---------------------------------------------------------------------

def plot_network(G, partition_map, title, fname):
    plt.figure(figsize=(9, 8))
    pos = nx.spring_layout(G, weight="weight", seed=RNG_SEED, k=0.6)
    communities = [partition_map[n] for n in G.nodes()]
    cmap = plt.cm.tab10
    weights = [G[u][v]["weight"] for u, v in G.edges()]
    nx.draw_networkx_edges(G, pos, alpha=0.25, width=[max(w, 0) * 2 for w in weights])
    nodes = nx.draw_networkx_nodes(
        G, pos, node_size=180, node_color=communities, cmap=cmap
    )
    nx.draw_networkx_labels(G, pos, font_size=6)
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, fname), dpi=200)
    plt.close()

plot_network(G_overall, partition, "Overall Opinion Network (colored by Louvain community)",
             "overall_network.png")

for t, g in topic_graphs.items():
    plot_network(g, topic_partitions[t], f"{t} Opinion Sub-Network", f"{t.lower()}_network.png")

# Degree distribution
plt.figure(figsize=(6, 4))
degrees = [d for _, d in G_overall.degree()]
plt.hist(degrees, bins=range(min(degrees), max(degrees) + 2), edgecolor="black")
plt.xlabel("Degree")
plt.ylabel("Number of respondents")
plt.title("Degree Distribution — Overall Opinion Network")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "degree_distribution.png"), dpi=200)
plt.close()

# Cross-topic ARI bar chart
plt.figure(figsize=(6, 4))
plt.bar(ari_results.keys(), ari_results.values(), color="#4C72B0")
plt.ylabel("Adjusted Rand Index vs. Overall Communities")
plt.title("How similar is each topic's opinion clustering to the overall network?")
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "cross_topic_ari.png"), dpi=200)
plt.close()

# ---------------------------------------------------------------------
# STEP 7 — EXPORT GRAPHS
# ---------------------------------------------------------------------

nx.write_graphml(G_overall, os.path.join(OUT_DIR, "overall_network.graphml"))
for t, g in topic_graphs.items():
    nx.write_graphml(g, os.path.join(OUT_DIR, f"{t.lower()}_network.graphml"))

# ---------------------------------------------------------------------
# STEP 8 — OPINION BRIDGE ANALYSIS
# ---------------------------------------------------------------------
# Which specific respondents structurally connect otherwise-separate
# opinion communities? High betweenness centrality identifies these
# "bridge" individuals. We additionally check how many distinct
# communities each bridge's direct neighbors belong to -- a bridge
# whose neighbors span many communities is a genuine cross-cutting
# opinion, not just a locally well-connected node.

bridges_df = centrality_df.sort_values("betweenness_centrality", ascending=False).head(5).copy()
neighbor_diversity = []
for rid in bridges_df["respondent_id"]:
    neighbors = list(G_overall.neighbors(rid))
    ndiv = centrality_df[centrality_df["respondent_id"].isin(neighbors)]["community"].nunique()
    neighbor_diversity.append(ndiv)
bridges_df["neighbor_community_diversity"] = neighbor_diversity
bridges_df.to_csv(os.path.join(OUT_DIR, "opinion_bridges.csv"), index=False)

print("\n[bridges] Top 5 opinion bridges (connect multiple communities):")
print(bridges_df[["respondent_id", "betweenness_centrality", "community",
                   "neighbor_community_diversity"]].to_string(index=False))

def plot_bridges(G, partition_map, bridge_ids, fname):
    plt.figure(figsize=(10, 9))
    pos = nx.spring_layout(G, weight="weight", seed=RNG_SEED, k=0.6)
    communities = [partition_map[n] for n in G.nodes()]
    node_sizes = [800 if n in bridge_ids else 150 for n in G.nodes()]
    edgecolors = ["red" if n in bridge_ids else "none" for n in G.nodes()]
    linewidths = [2.5 if n in bridge_ids else 0 for n in G.nodes()]
    nx.draw_networkx_edges(G, pos, alpha=0.2)
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=communities,
                            cmap=plt.cm.tab10, edgecolors=edgecolors, linewidths=linewidths)
    nx.draw_networkx_labels(G, pos, labels={n: n for n in bridge_ids}, font_size=9, font_weight="bold")
    plt.title("Opinion Bridges (red-ringed) — respondents connecting\ndifferent opinion communities, by betweenness centrality")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, fname), dpi=200)
    plt.close()

plot_bridges(G_overall, partition, set(bridges_df["respondent_id"]), "opinion_bridges.png")

# ---------------------------------------------------------------------
# STEP 9 — COMMUNITY OPINION ARCHETYPES
# ---------------------------------------------------------------------
# What does each opinion community actually *believe*, on average,
# relative to the class as a whole? For every community with >=4
# members (ignoring noise-sized micro-communities), compute its mean
# deviation from the overall class average on each topic, plus its
# 3 most distinctive individual survey items. This turns abstract
# "community 12" labels into interpretable opinion archetypes.

encoded["respondent_id"] = encoded["respondent_id"].astype(str)
encoded_with_comm = encoded.merge(
    centrality_df[["respondent_id", "community"]], on="respondent_id"
)
overall_item_mean = encoded[overall_cols].mean()
comm_sizes = encoded_with_comm["community"].value_counts()
big_communities = comm_sizes[comm_sizes >= 4].index.tolist()

archetype_rows = []
for comm in big_communities:
    sub = encoded_with_comm[encoded_with_comm["community"] == comm]
    comm_item_mean = sub[overall_cols].mean()
    item_deviation = comm_item_mean - overall_item_mean
    topic_dev = {
        t: item_deviation[cols].mean() for t, cols in topic_cols.items()
    }
    top_items = item_deviation.abs().sort_values(ascending=False).head(3).index.tolist()
    archetype_rows.append({
        "community": comm,
        "n_members": len(sub),
        **{f"{t}_deviation": round(v, 3) for t, v in topic_dev.items()},
        "most_distinctive_items": "; ".join(
            f"{it} ({item_deviation[it]:+.2f})" for it in top_items
        ),
    })

archetype_df = pd.DataFrame(archetype_rows).sort_values("n_members", ascending=False)
archetype_df.to_csv(os.path.join(OUT_DIR, "community_opinion_profiles.csv"), index=False)

print("\n[archetypes] Community opinion profiles (deviation from class average):")
print(archetype_df[["community", "n_members"] + [f"{t}_deviation" for t in TOPIC_MAP.values()]]
      .to_string(index=False))

plt.figure(figsize=(10, 6))
topics_list = list(TOPIC_MAP.values())
labels = [f"C{int(c)} (n={int(n)})" for c, n in zip(archetype_df["community"], archetype_df["n_members"])]
width = 0.2
x = range(len(archetype_df))
for i, t in enumerate(topics_list):
    plt.bar([p + i * width for p in x], archetype_df[f"{t}_deviation"], width=width, label=t)
plt.xticks([p + 1.5 * width for p in x], labels, rotation=20)
plt.axhline(0, color="black", linewidth=0.8)
plt.ylabel("Deviation from class-wide average opinion score")
plt.title("Opinion Archetypes: How Each Major Community Differs from the Class Average")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "community_opinion_profiles.png"), dpi=200)
plt.close()

print("\n[done] All outputs written to ./outputs and ./figures")

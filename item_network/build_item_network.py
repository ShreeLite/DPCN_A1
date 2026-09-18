"""
Item-Opinion Network Construction
==================================
Assignment 1: Opinion Network Formation

Nodes in this network are the 60 survey QUESTIONS (not respondents).
An edge between two items reflects how similarly respondents answered
them: if people who agreed strongly with item A also tended to agree
strongly with item B (and vice versa for disagreement), the two items
are opinion-correlated. This is the transpose of the person-opinion
network in opinion_network/build_network.py -- same underlying
methodology (Pearson correlation + mutual k-NN + Louvain), applied to
columns instead of rows -- so that the two analyses are directly
comparable.

This tells us which individual statements cluster together in people's
minds, independent of the topic label ("Technology", "Ethics/Society",
etc.) the survey design assigned them to.

Author: Kris Jain — Item-Opinion Network workstream
"""

import pandas as pd
import numpy as np
import networkx as nx
import community as community_louvain   # python-louvain
import matplotlib.pyplot as plt
from sklearn.metrics import adjusted_rand_score
import json
import os

OUT_DIR = "outputs"
FIG_DIR = "figures"
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

RNG_SEED = 42
np.random.seed(RNG_SEED)

# ---------------------------------------------------------------------
# STEP 1 — LOAD THE ALREADY-CLEANED RESPONSE MATRIX
# ---------------------------------------------------------------------
# Re-uses the cleaned/encoded matrix produced by the person-network
# pipeline (opinion_network/build_network.py) instead of re-implementing
# cleaning, so both networks are built from the exact same 87 respondents
# and -2..+2 encoding.

ENCODED_PATH = "../opinion_network/outputs/encoded_responses.csv"

df = pd.read_csv(ENCODED_PATH)
item_cols = [c for c in df.columns if c != "respondent_id"]
n_items = len(item_cols)

TOPIC_MAP = {"T": "Technology", "E": "Education", "S": "Ethics_Society", "V": "Environment"}
item_topic = {c: TOPIC_MAP[c[0]] for c in item_cols}
# Short codes (e.g. "T01") used as node labels in plots instead of full
# question text, which is far too long to render on a network diagram.
item_code = {c: c.split(".")[0] for c in item_cols}

print(f"[load] {df.shape[0]} respondents x {n_items} items.")

# ---------------------------------------------------------------------
# STEP 2 — ITEM-ITEM SIMILARITY (transpose of the person network)
# ---------------------------------------------------------------------
# Pearson correlation between two items' answer vectors across all 87
# respondents: do people who agree strongly with item A also tend to
# agree strongly with item B? Same similarity measure as the person
# network, for consistency and comparability, and for the same reason --
# it captures co-movement of opinion rather than raw response level.

corr = df[item_cols].corr().values  # n_items x n_items
np.fill_diagonal(corr, 1.0)

# ---------------------------------------------------------------------
# STEP 3 — GRAPH CONSTRUCTION (mutual k-NN sparsification)
# ---------------------------------------------------------------------
# Same sparsification strategy as the person network (mutual k-NN,
# k=8): an edge (i, j) is kept only if each item is among the other's
# top-8 most opinion-correlated items. This keeps only genuinely
# reciprocal, strong item associations and avoids a dense, uninterpretable
# fully-connected graph of 60x59/2 = 1770 possible edges.

K = 8

def build_knn_graph(sim_matrix, labels, k=K):
    n = len(labels)
    knn_sets = []
    for i in range(n):
        row = sim_matrix[i].copy()
        row[i] = -np.inf
        top_k = np.argsort(row)[-k:]
        knn_sets.append(set(top_k))

    G = nx.Graph(name="Item-Opinion Network")
    G.add_nodes_from(labels)
    for i in range(n):
        for j in knn_sets[i]:
            if i in knn_sets[j] and i < j:
                G.add_edge(labels[i], labels[j], weight=float(sim_matrix[i, j]))
    return G

G = build_knn_graph(corr, item_cols, k=K)
print(f"[graph] Item network: {G.number_of_nodes()} nodes, "
      f"{G.number_of_edges()} edges, density={nx.density(G):.3f}")

# ---------------------------------------------------------------------
# STEP 4 — METRICS (mirrors opinion_network/build_network.py Step 4)
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

metrics = graph_metrics(G)
print("[metrics]", metrics)

# ---------------------------------------------------------------------
# STEP 5 — COMMUNITY DETECTION
# ---------------------------------------------------------------------

partition = community_louvain.best_partition(G, weight="weight", random_state=RNG_SEED)
modularity = community_louvain.modularity(partition, G, weight="weight")
n_communities = len(set(partition.values()))
metrics["n_communities"] = n_communities
metrics["modularity"] = modularity
print(f"[community] Louvain found {n_communities} item communities, modularity={modularity:.3f}")

with open(os.path.join(OUT_DIR, "item_network_metrics.json"), "w") as f:
    json.dump(metrics, f, indent=2)

community_df = pd.DataFrame({
    "item": item_cols,
    "item_code": [item_code[c] for c in item_cols],
    "topic": [item_topic[c] for c in item_cols],
    "community": [partition[c] for c in item_cols],
    "degree": [G.degree(c) for c in item_cols],
})
community_df.to_csv(os.path.join(OUT_DIR, "item_community_assignments.csv"), index=False)

# ---------------------------------------------------------------------
# STEP 6 — DO ITEM COMMUNITIES RECOVER THE SURVEY'S TOPIC LABELS?
# ---------------------------------------------------------------------
# The survey was designed around 4 a-priori topics. Louvain community
# detection on opinion-correlation knows nothing about those labels. If
# communities line up closely with topics, opinions are organized the
# way the survey assumed. If not (low ARI), opinions cut across topic
# boundaries -- e.g. a "regulation & accountability" dimension that
# spans Technology and Ethics/Society items alike.

topic_labels = [item_topic[c] for c in item_cols]
community_labels = [partition[c] for c in item_cols]
ari_topic_vs_community = adjusted_rand_score(topic_labels, community_labels)
print(f"[ari] Item communities vs. a-priori survey topics: ARI = {ari_topic_vs_community:.3f}")

with open(os.path.join(OUT_DIR, "item_topic_ari.json"), "w") as f:
    json.dump({"ari_topic_vs_community": ari_topic_vs_community}, f, indent=2)

# Composition table: for each Louvain community, how many items from
# each of the 4 topics does it contain?
composition = (
    community_df.groupby(["community", "topic"]).size().unstack(fill_value=0)
)
composition["n_items"] = composition.sum(axis=1)
composition = composition.sort_values("n_items", ascending=False)
composition.to_csv(os.path.join(OUT_DIR, "item_community_topic_composition.csv"))
print("\n[composition] Item community x topic composition:")
print(composition.to_string())

# ---------------------------------------------------------------------
# STEP 7 — VISUALIZATIONS
# ---------------------------------------------------------------------

pos = nx.spring_layout(G, weight="weight", seed=RNG_SEED, k=0.7)

# (a) network colored by Louvain community
plt.figure(figsize=(10, 9))
communities = [partition[n] for n in G.nodes()]
weights = [G[u][v]["weight"] for u, v in G.edges()]
nx.draw_networkx_edges(G, pos, alpha=0.25, width=[max(w, 0) * 2.5 for w in weights])
nx.draw_networkx_nodes(G, pos, node_size=260, node_color=communities, cmap=plt.cm.tab10)
nx.draw_networkx_labels(G, pos, labels={n: item_code[n] for n in G.nodes()}, font_size=6.5)
plt.title(f"Item-Opinion Network (colored by Louvain community, k={K})\n"
          f"{n_communities} communities, modularity={modularity:.3f}")
plt.axis("off")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "item_network_by_community.png"), dpi=200)
plt.close()

# (b) same layout, colored by a-priori survey topic (for visual comparison)
plt.figure(figsize=(10, 9))
TOPIC_COLORS = {"Technology": "#4C72B0", "Education": "#DD8452",
                 "Ethics_Society": "#55A868", "Environment": "#C44E52"}
node_colors_by_topic = [TOPIC_COLORS[item_topic[n]] for n in G.nodes()]
nx.draw_networkx_edges(G, pos, alpha=0.25, width=[max(w, 0) * 2.5 for w in weights])
nx.draw_networkx_nodes(G, pos, node_size=260, node_color=node_colors_by_topic)
nx.draw_networkx_labels(G, pos, labels={n: item_code[n] for n in G.nodes()}, font_size=6.5)
handles = [plt.Line2D([0], [0], marker="o", color="w",
                       markerfacecolor=TOPIC_COLORS[t], markersize=10, label=t)
           for t in TOPIC_MAP.values()]
plt.legend(handles=handles, loc="lower left")
plt.title("Item-Opinion Network (colored by a-priori survey topic)\n"
          f"ARI vs. Louvain communities = {ari_topic_vs_community:.3f}")
plt.axis("off")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "item_network_by_topic.png"), dpi=200)
plt.close()

# (c) degree distribution
plt.figure(figsize=(6, 4))
degrees = [d for _, d in G.degree()]
plt.hist(degrees, bins=range(min(degrees), max(degrees) + 2), edgecolor="black", color="#4C72B0")
plt.xlabel("Degree")
plt.ylabel("Number of items")
plt.title("Degree Distribution — Item-Opinion Network")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "item_degree_distribution.png"), dpi=200)
plt.close()

# (d) community x topic composition heatmap
plt.figure(figsize=(7, 6))
comp_matrix = composition[list(TOPIC_MAP.values())]
plt.imshow(comp_matrix.values, cmap="Blues", aspect="auto")
plt.colorbar(label="Number of items")
plt.xticks(range(len(comp_matrix.columns)), comp_matrix.columns, rotation=20)
plt.yticks(range(len(comp_matrix.index)), [f"C{int(c)}" for c in comp_matrix.index])
plt.xlabel("A-priori survey topic")
plt.ylabel("Louvain item community")
plt.title("Item Community vs. Survey Topic Composition")
for i in range(comp_matrix.shape[0]):
    for j in range(comp_matrix.shape[1]):
        v = comp_matrix.values[i, j]
        if v > 0:
            plt.text(j, i, int(v), ha="center", va="center",
                      color="white" if v > comp_matrix.values.max() / 2 else "black")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "item_community_topic_heatmap.png"), dpi=200)
plt.close()

# ---------------------------------------------------------------------
# STEP 8 — EXPORT GRAPH
# ---------------------------------------------------------------------

nx.write_graphml(G, os.path.join(OUT_DIR, "item_network.graphml"))

print("\n[done] All outputs written to ./outputs and ./figures")

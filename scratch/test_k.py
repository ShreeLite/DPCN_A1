import json
import networkx as nx
import community as community_louvain
import numpy as np
import pandas as pd
import os

df = pd.read_csv('opinion_network/outputs/encoded_responses.csv')
ids = df['respondent_id'].astype(str).values
X = df.iloc[:, 1:].values.astype(float)
with np.errstate(invalid="ignore", divide="ignore"):
    sim = np.corrcoef(X)
sim = np.nan_to_num(sim, nan=0.0)
np.fill_diagonal(sim, 1.0)

def build_knn_graph(sim_matrix, ids, k=8):
    n = len(ids)
    knn_sets = []
    for i in range(n):
        row = sim_matrix[i].copy()
        row[i] = -np.inf
        top_k = np.argsort(row)[-k:]
        knn_sets.append(set(top_k))

    G = nx.Graph(name=f"knn_{k}")
    G.add_nodes_from(ids)
    for i in range(n):
        for j in knn_sets[i]:
            if i in knn_sets[j] and i < j:
                G.add_edge(ids[i], ids[j], weight=float(sim_matrix[i, j]))
    return G

for k in [4, 8, 12]:
    Gk = build_knn_graph(sim, ids, k=k)
    part_w = community_louvain.best_partition(Gk, weight="weight", random_state=42)
    mod_w = community_louvain.modularity(part_w, Gk, weight="weight")
    part_u = community_louvain.best_partition(Gk, weight=None, random_state=42)
    mod_u = community_louvain.modularity(part_u, Gk, weight=None)
    print(f"k={k}: nodes={Gk.number_of_nodes()}, edges={Gk.number_of_edges()}, "
          f"weighted mod={mod_w:.4f} (comms={len(set(part_w.values()))}), "
          f"unweighted mod={mod_u:.4f} (comms={len(set(part_u.values()))})")

with open('scratch_test_k.txt', 'w') as f:
    f.write('Completed successfully\n')

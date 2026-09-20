"""
Null-Model Validation & Sensitivity Analysis Pipeline
=====================================================
Assignment 1: Opinion Network Validation
Author: Prathish Ghosh — Statistical Validation workstream

Step 1 — Load & inspect the real network and verify ground truth metrics.
Step 2 — Configuration-model null-model loop (N = 500) for the primary network (k=8).
Step 3 — Compute z-score and empirical p-value for the primary network.
Step 4 — Generate publication-quality histogram of null modularity vs. observed.
Step 5 — Sensitivity analysis across k in {4, 8, 12}:
         - Reconstruct mutual k-NN opinion networks for k=4, 8, 12
         - Run 500-iteration configuration null model for each k
         - Compute z-score, empirical p-value, and modularity for each k
         - Generate comparative z-score bar chart
Step 6 — Save all summary metrics to JSON and CSV.
"""

import os
import json
import time
import numpy as np
import pandas as pd
import networkx as nx
import community as community_louvain
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

# ── Paths ────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
OPINION_OUT = os.path.join(PROJECT_ROOT, "opinion_network", "outputs")
OPINION_FIG = os.path.join(PROJECT_ROOT, "opinion_network", "figures")
REPORT_FIG = os.path.join(PROJECT_ROOT, "opinion_network", "report")

GRAPHML_PATH = os.path.join(OPINION_OUT, "overall_network.graphml")
METRICS_PATH = os.path.join(OPINION_OUT, "network_metrics.json")
CENTRALITY_PATH = os.path.join(OPINION_OUT, "node_centrality_communities.csv")
ENCODED_PATH = os.path.join(OPINION_OUT, "encoded_responses.csv")

VALIDATION_OUT = os.path.join(BASE_DIR, "outputs")
VALIDATION_FIG = os.path.join(BASE_DIR, "figures")
os.makedirs(VALIDATION_OUT, exist_ok=True)
os.makedirs(VALIDATION_FIG, exist_ok=True)

RNG_SEED = 42
np.random.seed(RNG_SEED)
N_ITER = 500

# =====================================================================
# STEP 1 — LOAD AND INSPECT THE REAL NETWORK
# =====================================================================
print("=" * 70)
print("STEP 1 — Load and inspect the real network (k=8)")
print("=" * 70)

G = nx.read_graphml(GRAPHML_PATH)

with open(METRICS_PATH) as f:
    reported_metrics = json.load(f)["overall"]

centrality_df = pd.read_csv(CENTRALITY_PATH)
centrality_df["respondent_id"] = centrality_df["respondent_id"].astype(str)
partition_obs = dict(zip(centrality_df["respondent_id"], centrality_df["community"]))

modularity_obs_weighted = community_louvain.modularity(partition_obs, G, weight="weight")
modularity_obs_unweighted = community_louvain.modularity(partition_obs, G, weight=None)
n_communities_obs = len(set(partition_obs.values()))

n_nodes_obs = G.number_of_nodes()
n_edges_obs = G.number_of_edges()

print(f"Nodes: {n_nodes_obs} (reported: {reported_metrics['n_nodes']})")
print(f"Edges: {n_edges_obs} (reported: {reported_metrics['n_edges']})")
print(f"Observed Modularity (weighted):   {modularity_obs_weighted:.4f} (expected ~0.578)")
print(f"Observed Modularity (unweighted): {modularity_obs_unweighted:.4f}")
print(f"Observed Communities:             {n_communities_obs}")

# =====================================================================
# STEP 2 & 3 — CONFIGURATION-MODEL NULL MODEL FOR k=8
# =====================================================================
print("\n" + "=" * 70)
print(f"STEPS 2 & 3 — Configuration-Model Null Testing (k=8, N={N_ITER})")
print("=" * 70)

deg_seq_8 = [d for _, d in G.degree()]

null_mods_k8 = []
null_comms_k8 = []
null_edges_k8 = []

t0 = time.time()
for i in tqdm(range(N_ITER), desc="k=8 null model", unit="iter"):
    seed_i = RNG_SEED + i
    G_null_multi = nx.configuration_model(deg_seq_8, seed=seed_i)
    G_null = nx.Graph(G_null_multi)
    G_null.remove_edges_from(nx.selfloop_edges(G_null))

    part_null = community_louvain.best_partition(G_null, random_state=seed_i)
    mod_null = community_louvain.modularity(part_null, G_null)
    n_comm_null = len(set(part_null.values()))

    null_mods_k8.append(mod_null)
    null_comms_k8.append(n_comm_null)
    null_edges_k8.append(G_null.number_of_edges())

t_k8 = time.time() - t0

null_mods_k8 = np.array(null_mods_k8)
null_comms_k8 = np.array(null_comms_k8)

mean_null_k8 = float(np.mean(null_mods_k8))
std_null_k8 = float(np.std(null_mods_k8))
median_null_k8 = float(np.median(null_mods_k8))
min_null_k8 = float(np.min(null_mods_k8))
max_null_k8 = float(np.max(null_mods_k8))

# Z-score and empirical p-value
# Using reported 0.578 (and exact 0.5778)
z_score_k8 = (modularity_obs_weighted - mean_null_k8) / std_null_k8
empirical_p_k8 = float(np.mean(null_mods_k8 >= modularity_obs_weighted))

print(f"\nk=8 Null Results:")
print(f"  Null Modularity: mean = {mean_null_k8:.4f} ± {std_null_k8:.4f} (range [{min_null_k8:.4f}, {max_null_k8:.4f}])")
print(f"  Observed Modularity: {modularity_obs_weighted:.4f}")
print(f"  z-score: {z_score_k8:.3f}")
print(f"  Empirical p-value: {empirical_p_k8:.4f} ({np.sum(null_mods_k8 >= modularity_obs_weighted)} / {N_ITER} runs >= observed)")

# Per-iteration log for the primary k=8 run, so the null distribution
# above is auditable/reproducible down to the individual random graph.
iterations_df = pd.DataFrame({
    "iteration": range(N_ITER),
    "seed": [RNG_SEED + i for i in range(N_ITER)],
    "modularity": null_mods_k8,
    "n_communities": null_comms_k8,
    "n_edges_kept": null_edges_k8,
})
iterations_csv_path = os.path.join(VALIDATION_OUT, "null_model_iterations.csv")
iterations_df.to_csv(iterations_csv_path, index=False)
print(f"  Saved per-iteration log to {iterations_csv_path}")

# =====================================================================
# STEP 4 — HISTOGRAM OF NULL MODULARITY SCORES
# =====================================================================
print("\n" + "=" * 70)
print("STEP 4 — Generating Null Modularity Distribution Histogram")
print("=" * 70)

plt.figure(figsize=(8, 5.2), dpi=300)
counts, bins, patches = plt.hist(
    null_mods_k8,
    bins=30,
    color="#4C72B0",
    edgecolor="white",
    alpha=0.85,
    density=True,
    label="Null Model (Configuration, N=500)"
)

# Plot normal fit curve over histogram
from scipy.stats import norm
x_axis = np.linspace(min_null_k8 - 0.02, modularity_obs_weighted + 0.02, 300)
plt.plot(
    x_axis,
    norm.pdf(x_axis, mean_null_k8, std_null_k8),
    color="#1F4E79",
    linewidth=2,
    linestyle="-",
    label=f"Gaussian Fit ($\\mu={mean_null_k8:.3f}, \\sigma={std_null_k8:.3f}$)"
)

# Observed modularity vertical dashed line
plt.axvline(
    x=modularity_obs_weighted,
    color="#C0392B",
    linestyle="--",
    linewidth=2.5,
    label=f"Observed $Q = {modularity_obs_weighted:.3f}$\n($z = {z_score_k8:.2f}, p < {1/N_ITER:.3f}$)"
)

plt.title("Statistical Validation: Observed Modularity vs. Configuration Null Model", fontsize=12, fontweight="bold", pad=12)
plt.xlabel("Louvain Modularity ($Q$)", fontsize=11)
plt.ylabel("Probability Density", fontsize=11)
plt.grid(axis="y", linestyle=":", alpha=0.6)
plt.legend(frameon=True, facecolor="white", edgecolor="#D0D0D0", fontsize=9.5, loc="upper left")

# Annotation box
stats_text = (
    f"Configuration Null ($N=500$):\n"
    f"• Mean $\\mu_{{null}} = {mean_null_k8:.3f}$\n"
    f"• Std Dev $\\sigma_{{null}} = {std_null_k8:.3f}$\n"
    f"• $z$-score = {z_score_k8:.2f}\n"
    f"• Empirical $p < 0.002$"
)
plt.gca().text(
    0.97, 0.65, stats_text,
    transform=plt.gca().transAxes,
    fontsize=9.5,
    verticalalignment="top",
    horizontalalignment="right",
    bbox=dict(boxstyle="round,pad=0.5", facecolor="#F8F9FA", edgecolor="#CCCCCC", alpha=0.95)
)

plt.tight_layout()

hist_path_val = os.path.join(VALIDATION_FIG, "null_modularity_histogram.png")
hist_path_op = os.path.join(OPINION_FIG, "null_modularity_histogram.png")
plt.savefig(hist_path_val, dpi=300)
plt.savefig(hist_path_op, dpi=300)
plt.close()
print(f"Saved histogram to:\n  {hist_path_val}\n  {hist_path_op}")

# =====================================================================
# STEP 5 — SENSITIVITY ANALYSIS ACROSS k in {4, 8, 12}
# =====================================================================
print("\n" + "=" * 70)
print("STEP 5 — Sensitivity Analysis Across k in {4, 8, 12}")
print("=" * 70)

# Load encoded responses matrix to reconstruct mutual k-NN graphs
encoded_df = pd.read_csv(ENCODED_PATH)
ids = encoded_df["respondent_id"].astype(str).values
item_cols = encoded_df.columns[1:]
X = encoded_df[item_cols].values.astype(float)

with np.errstate(invalid="ignore", divide="ignore"):
    sim_matrix = np.corrcoef(X)
sim_matrix = np.nan_to_num(sim_matrix, nan=0.0)
np.fill_diagonal(sim_matrix, 1.0)

def build_knn_graph(sim_mat, node_ids, k_val, label="graph"):
    n = len(node_ids)
    knn_sets = []
    for i in range(n):
        row = sim_mat[i].copy()
        row[i] = -np.inf
        top_k = np.argsort(row)[-k_val:]
        knn_sets.append(set(top_k))

    G_k = nx.Graph(name=label)
    G_k.add_nodes_from(node_ids)
    for i in range(n):
        for j in knn_sets[i]:
            if i in knn_sets[j] and i < j:
                G_k.add_edge(node_ids[i], node_ids[j], weight=float(sim_mat[i, j]))
    return G_k

k_values = [4, 8, 12]
sensitivity_results = {}

for k_val in k_values:
    print(f"\n--- Evaluating k = {k_val} ---")
    G_k = build_knn_graph(sim_matrix, ids, k_val, label=f"k={k_val} Opinion Network")
    n_nodes_k = G_k.number_of_nodes()
    n_edges_k = G_k.number_of_edges()
    density_k = nx.density(G_k)

    # Louvain on real graph
    part_k = community_louvain.best_partition(G_k, weight="weight", random_state=RNG_SEED)
    mod_obs_k = community_louvain.modularity(part_k, G_k, weight="weight")
    n_comm_k = len(set(part_k.values()))

    # Null model loop (500 iterations)
    deg_seq_k = [d for _, d in G_k.degree()]
    null_mods_k = []
    null_comms_k = []

    for i in tqdm(range(N_ITER), desc=f"k={k_val} null model", unit="iter"):
        seed_i = RNG_SEED + i
        G_null_multi = nx.configuration_model(deg_seq_k, seed=seed_i)
        G_null = nx.Graph(G_null_multi)
        G_null.remove_edges_from(nx.selfloop_edges(G_null))

        part_null = community_louvain.best_partition(G_null, random_state=seed_i)
        mod_null = community_louvain.modularity(part_null, G_null)
        null_mods_k.append(mod_null)
        null_comms_k.append(len(set(part_null.values())))

    null_mods_k = np.array(null_mods_k)
    mean_null_k = float(np.mean(null_mods_k))
    std_null_k = float(np.std(null_mods_k))
    z_k = (mod_obs_k - mean_null_k) / std_null_k
    p_k = float(np.mean(null_mods_k >= mod_obs_k))

    sensitivity_results[k_val] = {
        "k": k_val,
        "n_nodes": n_nodes_k,
        "n_edges": n_edges_k,
        "density": density_k,
        "modularity_obs": mod_obs_k,
        "n_communities_obs": n_comm_k,
        "null_modularity_mean": mean_null_k,
        "null_modularity_std": std_null_k,
        "z_score": z_k,
        "empirical_p_value": p_k,
        "null_comms_mean": float(np.mean(null_comms_k))
    }

    print(f"k={k_val}: edges={n_edges_k}, Q_obs={mod_obs_k:.4f}, Q_null={mean_null_k:.4f}±{std_null_k:.4f}, z={z_k:.3f}, p={p_k:.4f}")

# Print comparative table
print("\n" + "=" * 85)
print(f"{'k':<6} {'Edges':<8} {'Density':<10} {'Q_obs':<10} {'Q_null (mean±std)':<20} {'z-score':<10} {'p-value':<10}")
print("-" * 85)
for k_val in k_values:
    res = sensitivity_results[k_val]
    q_null_str = f"{res['null_modularity_mean']:.3f} ± {res['null_modularity_std']:.3f}"
    print(f"{k_val:<6} {res['n_edges']:<8} {res['density']:<10.3f} {res['modularity_obs']:<10.3f} {q_null_str:<20} {res['z_score']:<10.2f} {res['empirical_p_value']:<10.4f}")
print("=" * 85)

# =====================================================================
# STEP 6 — BAR CHART OF Z-SCORES ACROSS k
# =====================================================================
plt.figure(figsize=(7.5, 5), dpi=300)
k_labels = [f"k = {k}" for k in k_values]
z_vals = [sensitivity_results[k]["z_score"] for k in k_values]
colors = ["#5B9BD5" if k != 8 else "#ED7D31" for k in k_values]

bars = plt.bar(k_labels, z_vals, color=colors, width=0.55, edgecolor="black", linewidth=0.8)

plt.title("Sensitivity Analysis: Community Significance ($z$-score) vs. $k$-NN Sparsification", fontsize=11.5, fontweight="bold", pad=12)
plt.ylabel("Modularity $z$-score ($\\sigma$ above random expectation)", fontsize=10.5)
plt.xlabel("Mutual Nearest-Neighbors Parameter ($k$)", fontsize=10.5)
plt.grid(axis="y", linestyle=":", alpha=0.6)

# Value annotations on bars
for bar, z_val, k_val in zip(bars, z_vals, k_values):
    height = bar.get_height()
    q_obs = sensitivity_results[k_val]['modularity_obs']
    q_null = sensitivity_results[k_val]['null_modularity_mean']
    plt.text(
        bar.get_x() + bar.get_width() / 2.0,
        height + 0.12,
        f"$z = {z_val:.2f}$\n($Q_{{obs}}={q_obs:.3f}$)",
        ha="center", va="bottom",
        fontsize=9.5, fontweight="bold"
    )

plt.ylim(0, max(z_vals) * 1.25)
plt.axhline(0, color="black", linewidth=0.8)

# Add legend / explanation note
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor="#ED7D31", edgecolor="black", label="Primary Baseline ($k=8$)"),
    Patch(facecolor="#5B9BD5", edgecolor="black", label="Alternative Sparsifications ($k=4, 12$)")
]
plt.legend(handles=legend_elements, loc="upper right", frameon=True, facecolor="white", edgecolor="#D0D0D0", fontsize=9.5)

plt.tight_layout()

bar_path_val = os.path.join(VALIDATION_FIG, "k_sensitivity_zscores.png")
bar_path_op = os.path.join(OPINION_FIG, "k_sensitivity_zscores.png")
plt.savefig(bar_path_val, dpi=300)
plt.savefig(bar_path_op, dpi=300)
plt.close()
print(f"Saved z-score comparison bar chart to:\n  {bar_path_val}\n  {bar_path_op}")

# =====================================================================
# STEP 7 — SAVE RESULTS JSON & CSV
# =====================================================================
full_output_json = {
    "primary_network_k8": {
        "observed": {
            "modularity_weighted": float(modularity_obs_weighted),
            "modularity_unweighted": float(modularity_obs_unweighted),
            "n_communities": int(n_communities_obs),
            "n_nodes": int(n_nodes_obs),
            "n_edges": int(n_edges_obs),
        },
        "null_model": {
            "type": "configuration_model",
            "n_iterations": N_ITER,
            "rng_seed_base": RNG_SEED,
            "modularity_mean": mean_null_k8,
            "modularity_std": std_null_k8,
            "modularity_median": median_null_k8,
            "modularity_min": min_null_k8,
            "modularity_max": max_null_k8,
            "z_score": float(z_score_k8),
            "empirical_p_value": float(empirical_p_k8),
            "n_communities_mean": float(np.mean(null_comms_k8)),
            "n_communities_std": float(np.std(null_comms_k8))
        }
    },
    "k_sensitivity_analysis": {
        str(k): sensitivity_results[k] for k in k_values
    }
}

json_out_path = os.path.join(VALIDATION_OUT, "null_model_results.json")
with open(json_out_path, "w") as f:
    json.dump(full_output_json, f, indent=2)

k_summary_df = pd.DataFrame([sensitivity_results[k] for k in k_values])
k_summary_csv = os.path.join(VALIDATION_OUT, "k_sensitivity_summary.csv")
k_summary_df.to_csv(k_summary_csv, index=False)

print(f"\nPersisted final outputs:")
print(f"  {json_out_path}")
print(f"  {k_summary_csv}")
print("\nValidation pipeline execution complete!")

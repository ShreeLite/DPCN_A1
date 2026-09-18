"""
Dataset Documentation — Descriptive Statistics
================================================
Assignment 1: Opinion Network Formation

This script extends the Dataset Documentation section of the report.
It loads the already-cleaned/encoded response matrix produced by
Shreem's build_network.py (opinion_network/outputs/encoded_responses.csv)
and computes descriptive statistics and distribution plots per topic,
used to characterize the survey data before any network is built.

Author: Kris Jain — Dataset Documentation workstream
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

OUT_DIR = "outputs"
FIG_DIR = "figures"
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

ENCODED_PATH = "../opinion_network/outputs/encoded_responses.csv"

df = pd.read_csv(ENCODED_PATH)
question_cols = [c for c in df.columns if c != "respondent_id"]

TOPIC_MAP = {"T": "Technology", "E": "Education", "S": "Ethics_Society", "V": "Environment"}
col_topic = {c: TOPIC_MAP[c[0]] for c in question_cols}
topic_cols = {t: [c for c in question_cols if col_topic[c] == t] for t in TOPIC_MAP.values()}

# ---------------------------------------------------------------------
# Per-topic descriptive statistics
# ---------------------------------------------------------------------
# For each topic, flatten all (respondent x item) values into one long
# vector and summarize it. Mean indicates overall agreement/disagreement
# lean (-2..+2 scale); std indicates how spread out / contested opinion
# is within that topic.

rows = []
long_form = {}
for t, cols in topic_cols.items():
    vals = df[cols].values.flatten()
    long_form[t] = vals
    rows.append({
        "topic": t,
        "n_items": len(cols),
        "n_values": len(vals),
        "mean": np.mean(vals),
        "std": np.std(vals, ddof=1),
        "median": np.median(vals),
        "pct_agree_or_above": np.mean(vals > 0) * 100,   # Agree or Strongly Agree
        "pct_disagree_or_below": np.mean(vals < 0) * 100,  # Disagree or Strongly Disagree
    })

topic_stats = pd.DataFrame(rows).sort_values("mean", ascending=False)
topic_stats.to_csv(os.path.join(OUT_DIR, "topic_descriptive_stats.csv"), index=False)
print("[stats] Per-topic descriptive statistics:")
print(topic_stats.round(3).to_string(index=False))

# Per-item statistics (all 60 items), used to find the most contested /
# most consensual individual questions.
item_rows = []
for c in question_cols:
    vals = df[c].values
    item_rows.append({
        "item": c,
        "topic": col_topic[c],
        "mean": np.mean(vals),
        "std": np.std(vals, ddof=1),
    })
item_stats = pd.DataFrame(item_rows)
item_stats.to_csv(os.path.join(OUT_DIR, "item_descriptive_stats.csv"), index=False)

most_consensual = item_stats.sort_values("std").head(5)
most_contested = item_stats.sort_values("std", ascending=False).head(5)
print("\n[stats] 5 most consensual items (lowest std):")
print(most_consensual[["item", "topic", "mean", "std"]].round(3).to_string(index=False))
print("\n[stats] 5 most contested items (highest std):")
print(most_contested[["item", "topic", "mean", "std"]].round(3).to_string(index=False))

# ---------------------------------------------------------------------
# Plot 1 — distribution of responses per topic (boxplot)
# ---------------------------------------------------------------------
plt.figure(figsize=(7, 5))
topics_list = list(TOPIC_MAP.values())
data = [long_form[t] for t in topics_list]
bp = plt.boxplot(data, tick_labels=topics_list, patch_artist=True, showmeans=True)
colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
for patch, color in zip(bp["boxes"], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)
plt.axhline(0, color="black", linewidth=0.8, linestyle="--")
plt.ylabel("Encoded response (-2 = Strongly Disagree .. +2 = Strongly Agree)")
plt.title("Distribution of Responses by Topic")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "topic_response_boxplot.png"), dpi=200)
plt.close()

# ---------------------------------------------------------------------
# Plot 2 — histogram of per-item mean opinion, faceted by topic
# ---------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(9, 7), sharex=True, sharey=True)
bins = np.linspace(-2, 2, 17)
for ax, t, color in zip(axes.flat, topics_list, colors):
    means = item_stats[item_stats["topic"] == t]["mean"]
    ax.hist(means, bins=bins, color=color, edgecolor="black", alpha=0.8)
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_title(t)
    ax.set_xlabel("Item mean response")
    ax.set_ylabel("Number of items")
fig.suptitle("Distribution of Item-Mean Opinion, by Topic")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "item_mean_distribution.png"), dpi=200)
plt.close()

# ---------------------------------------------------------------------
# Plot 3 — overall response distribution (all 87*60 values)
# ---------------------------------------------------------------------
plt.figure(figsize=(6, 4))
all_vals = df[question_cols].values.flatten()
plt.hist(all_vals, bins=np.linspace(-2, 2, 17), color="#4C72B0", edgecolor="black")
plt.xlabel("Encoded response")
plt.ylabel("Count")
plt.title("Overall Distribution of All Encoded Responses\n(87 respondents x 60 items)")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "overall_response_distribution.png"), dpi=200)
plt.close()

print("\n[done] Descriptive statistics and figures written to ./outputs and ./figures")

# Opinion Network Formation — Network Construction Pipeline

This directory contains the **network construction** component of our
Assignment 1 submission. It turns raw survey responses into a graph and
runs the analysis needed for the report's "Pipeline Followed",
"Analysis and Visualizations", and part of "Results and Discussion"
sections.

## What this does

1. **Cleaning** — drops respondents who left >50% of the survey blank
   (9 of 96 respondents), and treats `"No Comments"` as a missing value
   rather than a neutral opinion.
2. **Encoding** — maps the 5-point Likert scale to an ordinal numeric
   scale (`Strongly Disagree=-2 ... Strongly Agree=+2`), then fills any
   remaining sparse gaps with that item's class-average response.
3. **Similarity** — computes **Pearson correlation** between every pair
   of respondents' opinion profiles (both overall, across all 60 items,
   and separately per topic: Technology / Education / Ethics-Society /
   Environment). Correlation was chosen over cosine/Euclidean distance
   because it captures whether two people's opinions move together across
   statements, independent of one person tending to answer more
   extremely than another.
4. **Graph construction** — sparsifies the dense similarity matrix into
   a graph using **mutual k-Nearest-Neighbours** (k=8): an edge is kept
   only if each respondent is among the other's top-8 most similar peers.
   This avoids a meaningless fully-connected graph and keeps only
   reciprocal, strong opinion-closeness relationships.
5. **Metrics** — degree, density, clustering coefficient, average path
   length / diameter (on the largest connected component), computed for
   the overall network and all 4 topic sub-networks.
6. **Community detection** — Louvain modularity maximization on the
   overall network, plus separate community detection on each topic
   sub-network.
7. **Cross-topic comparison** — Adjusted Rand Index (ARI) between each
   topic's community partition and the overall partition, to answer:
   *"does opinion clustering on this topic match how people cluster
   overall?"*
8. **Visualization** — force-directed layouts (colored by community),
   degree distribution histogram, and an ARI comparison bar chart.
9. **Opinion bridge analysis** — identifies the specific respondents with
   the highest betweenness centrality (structural "bridges") and checks
   how many distinct communities each bridge's direct neighbors span,
   distinguishing genuine cross-cutting opinions from locally
   well-connected nodes.
10. **Community opinion archetypes** — for every community with 4+
    members, computes its average deviation from the class-wide opinion
    on each topic, plus its 3 most distinctive individual survey items,
    turning abstract community IDs into interpretable opinion profiles
    (e.g. "skeptic cluster", "strong-consensus cluster").

## How to run

```bash
pip install pandas numpy networkx python-louvain scikit-learn matplotlib
python build_network.py
```

Outputs are written to `outputs/` (data files, `.graphml` graphs,
metrics as JSON/CSV) and `figures/` (all plots used in the report).

## Key design choices (for the "Pipeline Followed" report section)

| Choice | Alternative considered | Why we chose this |
|---|---|---|
| Pearson correlation similarity | Cosine similarity, Euclidean distance | Robust to individual response-scale bias (someone who always answers more strongly) |
| Mutual k-NN sparsification (k=8) | Fixed similarity threshold | Guarantees every node has a bounded, comparable degree regardless of how "central" their opinions are; avoids threshold tuning |
| Louvain community detection | Girvan-Newman, spectral clustering | Standard for weighted, sparse social networks; fast and well-supported (`python-louvain`) |
| Drop respondents with >50% missing | Impute everything | Data integrity — a respondent with mostly blank answers can't be meaningfully placed in an opinion network |

## Findings worth highlighting in Results & Discussion

- The overall network fragments into **~25 Louvain communities** at
  modularity ≈0.58 — opinion similarity is quite fine-grained, not just
  2–3 broad "camps".
- **Education and Technology** opinions correlate most with a
  respondent's overall community membership (ARI 0.18 and 0.10) — how
  someone feels about these topics is a decent predictor of their
  broader opinion "cluster".
- **Ethics/Society and Environment** opinions barely correlate with
  overall clustering (ARI ≈0.02–0.03) — these are areas of much broader,
  cross-cutting consensus/dissent that doesn't map onto the same social
  groupings, i.e. people who disagree about tech/education still tend to
  converge (or diverge) similarly on environmental and ethical questions.
- **Opinion bridges**: a handful of specific respondents (e.g. #81)
  have neighbors spanning up to 5 distinct communities — the class's
  opinion diversity is held together less by broad consensus and more
  by a small number of structurally central "bridge" individuals.
- **Opinion archetypes**: e.g. community 0 (9 members) is a clear
  "skeptic" cluster, scoring well below the class average on both
  Ethics/Society and Environment items, while community 16 (6 members)
  is a "strong-consensus" cluster scoring above average across every
  topic.

## Files

- `build_network.py` — the full pipeline, run top to bottom
- `outputs/encoded_responses.csv` — cleaned, numeric-encoded response matrix
- `outputs/*.graphml` — overall + 4 topic networks, importable into Gephi
- `outputs/network_metrics.json` — all computed graph metrics
- `outputs/node_centrality_communities.csv` — per-respondent centrality + community assignment
- `outputs/cross_topic_ari.json` — ARI comparison numbers
- `outputs/opinion_bridges.csv` — top 5 bridge respondents + neighbor community diversity
- `outputs/community_opinion_profiles.csv` — per-community opinion archetype profiles
- `figures/*.png` — all plots, including `opinion_bridges.png` and `community_opinion_profiles.png`

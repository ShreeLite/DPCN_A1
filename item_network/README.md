# Item-Opinion Network — Which Questions Cluster Together?

This directory contains the **item-opinion network** component of our
Assignment 1 submission (Kris's workstream). It answers a different
question from the person-opinion network in `../opinion_network/`:
instead of asking *which respondents think alike*, it asks
*which survey questions get answered alike* — i.e., do people who
agree with "AI should be regulated" also tend to agree with
"protect personal data", regardless of who they are?

## What this does

1. **Reuses** the cleaned, encoded response matrix already produced by
   `../opinion_network/build_network.py`
   (`../opinion_network/outputs/encoded_responses.csv`) — no re-cleaning,
   so both networks are built from the exact same 87 respondents.
2. **Similarity** — computes the Pearson correlation between every pair
   of the 60 survey *items* (columns), across all 87 respondents. This
   is the transpose of the person-network's similarity step: correlate
   columns instead of rows.
3. **Graph construction** — sparsifies the dense 60x60 correlation
   matrix using the same **mutual k-Nearest-Neighbours** procedure
   (k=8) as the person network, for methodological consistency.
4. **Metrics** — node/edge counts, density, clustering coefficient,
   average path length/diameter on the largest component.
5. **Community detection** — Louvain modularity maximization on the
   item network, producing groups of survey questions that people tend
   to answer similarly.
6. **Topic-recovery check** — compares the discovered item communities
   against the survey's own 4 a-priori topic labels (Technology /
   Education / Ethics-Society / Environment) using Adjusted Rand Index.
   A low ARI means opinions are organized differently from how the
   survey was designed — i.e., real "opinion dimensions" cut across the
   nominal topic boundaries.
7. **Visualization** — the item network colored two ways (by discovered
   Louvain community, and by a-priori survey topic) on the *same*
   spring layout for direct visual comparison, a degree-distribution
   histogram, and a community x topic composition heatmap.

## How to run

```bash
pip install pandas numpy networkx python-louvain scikit-learn matplotlib
python build_item_network.py
```

Must be run with `../opinion_network/outputs/encoded_responses.csv`
present (i.e. after `../opinion_network/build_network.py` has been run
at least once). Outputs go to `outputs/` and `figures/`.

## Key design choices

| Choice | Rationale |
|---|---|
| Reuse Shreem's cleaned matrix rather than re-cleaning the raw survey | Guarantees the person- and item-networks describe the same 87 respondents; avoids duplicated/possibly-inconsistent cleaning logic |
| Pearson correlation between items (not respondents) | Direct transpose of the person-network similarity measure — same rationale (invariant to a respondent's overall tendency to answer extremely) |
| Mutual k-NN, k=8 (same k as person network) | Methodological consistency with `../opinion_network/`, enabling direct comparison of network densities/structure between the two workstreams |
| ARI against a-priori topic labels | The survey's 4 topics are a *design* assumption, not a measured fact; testing whether Louvain communities recover them tells us whether people's real opinion structure matches the survey's categorization |

## Findings worth highlighting in Results & Discussion

- The item network is sparser and less cleanly separated than the
  person network: Louvain finds **9 communities** at
  **modularity ≈ 0.43**, but 3 of these are singleton/near-singleton
  items with degree 0–1 (items nobody's opinion pattern strongly tracks).
- **ARI between item communities and a-priori survey topics ≈ 0.14** —
  low. Item communities do **not** simply reproduce the 4 survey topics.
- Instead, the two largest communities each **span two nominal topics**:
  one groups Technology + Education items (a "modernization /
  innovation-attitude" cluster), and another groups Ethics/Society +
  Environment items (a "collective responsibility" cluster). This
  suggests the class's opinion space has (at least) two broad
  cross-cutting dimensions that the survey's 4-topic design doesn't
  directly capture.

## Files

- `build_item_network.py` — the full item-network pipeline, run top to bottom
- `outputs/item_network_metrics.json` — graph metrics + community count/modularity
- `outputs/item_community_assignments.csv` — per-item community, topic, degree
- `outputs/item_topic_ari.json` — ARI between item communities and survey topics
- `outputs/item_community_topic_composition.csv` — community x topic item counts
- `outputs/item_network.graphml` — the item network, importable into Gephi
- `figures/item_network_by_community.png` — network colored by Louvain community
- `figures/item_network_by_topic.png` — same layout, colored by a-priori topic
- `figures/item_degree_distribution.png` — degree histogram
- `figures/item_community_topic_heatmap.png` — community x topic composition heatmap

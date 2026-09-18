# Dataset Documentation — Descriptive Statistics

This directory contains the extra descriptive-statistics material Kris
added on top of Shreem's cleaning/encoding write-up, for the "Dataset
Documentation" section of the report.

## What this does

Loads `../opinion_network/outputs/encoded_responses.csv` (the already
cleaned, -2..+2 encoded response matrix for 87 respondents x 60 items)
and computes:

1. **Per-topic descriptive statistics** — mean, standard deviation,
   median, and % agreeing/disagreeing, computed over all (respondent,
   item) values within each of the 4 topics.
2. **Per-item descriptive statistics** — mean/std for each of the 60
   individual items, used to find the most consensual and most
   contested statements in the survey.
3. **Distribution plots**:
   - a boxplot comparing response distributions across the 4 topics,
   - a 2x2 histogram grid of per-item mean opinion, faceted by topic,
   - a histogram of the entire encoded response matrix (all 5,220
     respondent x item values), showing the overall skew of the survey.

## How to run

```bash
pip install pandas numpy matplotlib
python descriptive_stats.py
```

Must be run after `../opinion_network/build_network.py` has produced
`../opinion_network/outputs/encoded_responses.csv`. Outputs go to
`outputs/` and `figures/`.

## Findings worth highlighting in Dataset Documentation

- The class agrees with almost every statement on average: topic means
  are all positive (Environment 1.39, Ethics/Society 1.28, Technology
  0.99, Education 0.91 on the -2..+2 scale), and roughly 90% of
  Environment responses and 86% of Ethics/Society responses are
  "Agree" or "Strongly Agree".
- Education has both the lowest mean (0.91) and the highest spread
  (std 1.15) — it's the topic with the most genuine disagreement.
- The single most contested items are all Education items about
  assessment/attendance policy (e.g. "online learning can effectively
  complement classroom teaching", "class attendance should be
  compulsory") — plausibly, students' personal experience varies more
  here than for abstract tech/ethics/environment statements.
- The most consensual items are broad, low-stakes-sounding statements
  ("continuous learning is essential", "companies should be
  accountable for environmental impact") — the kind of item a
  self-report survey struggles to get real disagreement on.
- This pattern is consistent with **social-desirability bias**: many
  items are phrased as normatively "good" positions (sustainability,
  ethics, accountability, personal responsibility), which respondents
  are inclined to agree with regardless of their true attitude,
  especially in a non-anonymous or peer-visible class survey context.
  Combined with **acquiescence bias** (a general tendency to agree with
  presented statements) and possible **self-selection** in who chose
  to respond at all, the strong rightward skew likely overstates true
  consensus, and correlational structure (as used throughout both the
  person- and item-networks) is a more robust signal than raw response
  levels.

## Files

- `descriptive_stats.py` — the descriptive-statistics script
- `outputs/topic_descriptive_stats.csv` — per-topic mean/std/median/% agree
- `outputs/item_descriptive_stats.csv` — per-item mean/std (all 60 items)
- `figures/topic_response_boxplot.png` — response distribution by topic
- `figures/item_mean_distribution.png` — item-mean histograms, faceted by topic
- `figures/overall_response_distribution.png` — overall response histogram

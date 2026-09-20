# DPCN Assignment 1 — Opinion Network Formation

Team: **Complex Network of Dynamic People**

We build and analyze three related networks from a class survey (60 Likert-scale
questions across Technology, Education, Ethics/Society, and Environment, 87 usable
respondents after cleaning): a **person-opinion network**, an **item-opinion network**,
and a **statistical validation** of the person-network's community structure. The
final report is at [`opinion_network/report/report.pdf`](opinion_network/report/report.pdf)
(source: [`report.tex`](opinion_network/report/report.tex)).

## Structure

| Folder | Owner | What it does |
|---|---|---|
| [`opinion_network/`](opinion_network) | Shreem | Cleans the raw survey, builds the **person-opinion network** (nodes = respondents, mutual $k$-NN on Pearson correlation, Louvain communities), and contains the report. |
| [`item_network/`](item_network) | Kris | Builds the **item-opinion network** (nodes = the 60 survey questions instead of people) and compares it against the survey's 4 a-priori topics. |
| [`dataset_documentation/`](dataset_documentation) | Kris | Per-topic descriptive statistics, distribution plots, and a self-report survey-bias discussion, used in the report's Dataset Documentation section. |
| [`validation/`](validation) | Prathish | Statistical validation of the person-network's community structure via a configuration-model null test (500 iterations), plus a $k \in \{4,8,12\}$ sensitivity analysis. |

Each of the four pipeline folders has its own `README.md` with exact run instructions,
dependencies, and design-choice rationale.

## Running the full pipeline

Each script writes into its own `outputs/` and `figures/` folders (already committed,
so you don't need to rerun anything to read the report). To reproduce everything from
scratch, run in this order — later stages depend on earlier ones' `outputs/`:

```bash
pip install pandas numpy networkx python-louvain scikit-learn matplotlib scipy tqdm

cd opinion_network      && python3 build_network.py          # 1. person network (writes encoded_responses.csv)
cd ../item_network       && python3 build_item_network.py     # 2. item network (needs step 1's output)
cd ../dataset_documentation && python3 descriptive_stats.py   # 3. descriptive stats (needs step 1's output)
cd ../validation         && python3 build_null_model.py       # 4. null-model validation (needs step 1's output)
```

`opinion_network/build_network.py` reads the raw survey from `../Survey_Results_UC.csv`
(repo root) — everything downstream reads only the already-cleaned
`opinion_network/outputs/encoded_responses.csv`, so steps 2–4 don't need the raw CSV.

# Validation — Null-Model Testing

This folder contains the **null-model validation pipeline** for the opinion network analysis. It tests whether the observed community structure (modularity, community count) is statistically significant — i.e., not an artifact of the network's degree sequence or density alone.

## Structure

```
validation/
├── build_null_model.py   # Main pipeline script
├── figures/              # Generated plots (modularity distributions, etc.)
├── outputs/              # Null-model results (JSON, CSV)
└── README.md             # This file
```

## Dependencies

- `networkx`
- `python-louvain` (import as `community`)
- `numpy`
- `pandas`
- `matplotlib`
- `scipy`

## Usage

```bash
cd DPCN_A1/validation
python3 build_null_model.py
```

## Pipeline Steps

1. **Load & inspect** the real network (`overall_network.graphml`) and cross-check against reported metrics in `network_metrics.json`.

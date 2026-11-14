# HydroGraph-FL

**Hydrology-Aware Federated Spatiotemporal Graph Learning for River Water Quality Forecasting**

Python research repository with **original CSV datasets included**, data cleaning, observed-data forecasting baselines, a single-monitor GRU prototype and 10-fold forward-chaining evaluation, plus a multi-station GRU–directed-GNN/FedAvg/FedProx/QDE research implementation.

**Scope:** The Ballycanew data are real field observations from a single monitored catchment; the separate five-client directed river-network benchmark is computationally simulated. Neural code availability is not evidence that manuscript Ridge benchmark numbers came from the full neural model. **No claim of 97% field accuracy is made.**

## Quick start

```bash
python -m pip install -r requirements.txt
python scripts/build_ballycanew.py
python scripts/train_observed_baselines.py
python scripts/train_observed_gru.py
python scripts/cross_validate_observed_gru.py
python run_neural.py --mode central --quick
python -m unittest discover -s tests -v
```

Training the full simulated-network model (longer experiments):

```bash
python run_neural.py --mode central
python run_neural.py --mode fedavg
python run_neural.py --mode fedprox
python run_neural.py --mode qde
python run_neural.py --mode all
```

`run_neural.py` defaults to `data/benchmark`; its outputs are placed in `outputs/neural/`. Consult `configs/default.json` for epochs/rounds and QDE search budget. `--quick` runs a smoke test; **do not cite quick-test metrics as definitive experiments**.

## Directory layout

```text
HydroGraph-FL_GitHub_Complete/
├── data/
│   ├── observed/raw/          # Original Ballycanew CSV files
│   ├── observed/processed/    # Daily merged and supervised CSVs
│   └── benchmark/             # Separately identified five-client computational CSVs
├── scripts/
│   ├── build_ballycanew.py
│   ├── train_observed_baselines.py
│   ├── train_observed_gru.py
│   ├── cross_validate_observed_gru.py
│   └── download_field_data.py
├── src/hydrograph.py          # Directed GRU-GNN / masked physical loss / federation / QDE
├── run_neural.py
├── configs/default.json
├── tests/
├── outputs/observed/         # Recorded observational histories and example metrics
├── outputs/benchmark/        # Recorded legacy benchmark evaluation CSVs
├── requirements.txt
└── DATA_PROVENANCE.md
```

## Data split
Observed Ballycanew one-day-ahead supervised data use **2009–2014 train (1,838)**, **2015 validation (365)** and **2016 independent test (365)**. Standardization and the turbidity alert threshold are fitted on training data only. The 10-fold approach uses chronological expanding windows; random shuffling across the timeline is not an acceptable replacement.

## Outputs and reproducibility
Run baseline and neural scripts from the repository root. Model scripts write metrics, predictions, checkpoint files and convergence histories under `outputs/`. To regenerate benchmark Ridge tables run the separately supplied legacy benchmark procedure; the benchmark CSVs and recorded outputs here allow table/figure verification. See `DATA_PROVENANCE.md` for limitations, licensing, and interpretation.

## Citation and data source
Negri C, Mellander P-E, Schurch N, Wade AJ, Gagkas Z, Wardell-Johnson DH, Adams K, Glendell M (2024). Bayesian network modelling of phosphorus pollution in agricultural catchments with high-resolution data. *Environmental Modelling & Software* 178:106073. https://doi.org/10.1016/j.envsoft.2024.106073

Research repo destination: https://github.com/Qaisar256/HydroGraph-FL

**Before public upload:** verify the Ballycanew data reuse permissions and source measurement units; remove any materials without redistribution authorization.

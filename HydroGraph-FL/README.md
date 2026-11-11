# HydroGraph-FL neural research implementation

**This is runnable PyTorch code for a proposed GRU + directed edge-aware GNN architecture, with FedAvg/FedProx federated training, optional masked physical residual and classical quantum-inspired differential evolution (QDE).** It includes the computer-generated data and the previous separate linear surrogate benchmark.

## Data and scientific scope

The delivered `data/*.csv` are **computer-generated**. There are five independent artificial catchments, seven stations per catchment, 540 days per station (18,900 daily records) and 30 directed upstream-to-downstream edges. Original `forecast_supervised_rows.csv` contains the chronological train/validation/test assignments and one-day-ahead labels. It is **not** WQP/USGS field data. The new PyTorch project is independent of the previously reported linear surrogate benchmark. Running the code creates genuine new neural evaluation numbers. Earlier manuscript figures such as **97.15% / MAE 0.947** were illustrative and are not validated by the presence of this code.

The temporal GRU processes each station's 14-day sequence; directed message passing aggregates only upstream neighbors using a learned edge-conditioned gate and discharge feature; a shared regression head predicts next-day turbidity at all seven stations. Five independent catchments are federated clients. FedAvg and FedProx aggregate parameter states weighted by client sample counts. No client station observations are exchanged during federated fitting.

**Physical consistency limitation:** The included CSVs do not contain independently observed sediment mass concentration, source/sink rates, storage change, reach volumes, or measured travel times. Turbidity in NTU is not mass concentration. Therefore, physics regularization is masked off (`valid_physics_constraints=0`) in this benchmark. `physics_flux_loss()` is an extendable template that activates only for explicitly supplied, correctly unit-checked flux terms. To activate a physically interpretable sediment balance, provide an `observed_physics_inputs.csv` file matching the supplied header-only template, containing observed upstream sediment fluxes, downstream discharge, net sources/sinks, storage changes, site-specific measured turbidity-to-SSC calibration, a physical residual scale, and `valid=1` only for physically supported records. The implemented loader uses these values in a masked sediment-flux loss in kg/s. Input day is the forecast anchor day; supplied source/storage terms must be forward-available forecasts or independent exogenous inputs, never future measurements leaked into training. Setting `physics_weight` to a nonzero value alone **does not** assert physics-guided training on these data.

## Installation and execution

Python 3.10+ recommended. In this extracted folder:

```bash
python -m pip install -r requirements.txt
python run_neural.py --mode central --quick      # 1 epoch smoke test
python run_neural.py --mode central             # train GRU-GNN centrally
python run_neural.py --mode fedavg              # five-client federated GRU-GNN
python run_neural.py --mode fedprox             # FedProx federated GRU-GNN
python run_neural.py --mode qde                 # QDE search then GRU-GNN training
python run_neural.py --mode all                 # run every experiment
python -m unittest discover -s tests -v
```

Use `--device cuda` if CUDA-enabled PyTorch is available. Modify `configs/default.json` to change model width, GRU/GNN layer counts, local epochs and search budgets.

## Output files

- `outputs/neural/run_provenance.json`: seed, training, validation, test counts, alert threshold, and explicit physical-residual availability.
- `outputs/neural/neural_comparison_metrics.csv`: test MAE, RMSE, alert accuracy, F1, ROC AUC, and validation MAE, **calculated from actual predictions**.
- `outputs/neural/<mode>_test_predictions.csv`: station-day targets and predictions.
- `outputs/neural/*_training_history.csv`: epoch/round level validation trace.
- `outputs/neural/*.pt`: saved best-validation PyTorch model checkpoints.
- `outputs/neural/qde_search.csv`, `qde_best_config.json`: objective evaluations, trial configuration, and selected QDE hyperparameters.

**QDE disclosure:** The optimizer uses bounded probability-amplitude-inspired angle encoding `sin²(θ)` plus DE mutation/crossover, candidate selection, and angular rotation toward the best candidate. It runs on an ordinary CPU/GPU and uses no quantum hardware. The optimization objective is **validation MAE**, never test error.

## Reproducibility and evaluation safeguards

1. Split memberships are inherited from the original supervised CSV and never randomly mixed across time. Each sequence uses present/past data only; future target day is excluded from the input.
2. Normalization fits training-period records only. The binary alert threshold is the training-target 75th percentile.
3. Training/validation selection is separate from the held-out test set. Test errors are computed only after final model selection.
4. FedAvg/FedProx use the five catchments as clients. This **does not** simulate cryptographic secure aggregation or privacy guarantees.
5. No observed river skill, verified hydrological conservation, or QDE superiority is claimed without corresponding evaluations.

## Original surrogate benchmark

`src/legacy_surrogate_benchmark.py` preserves the earlier linear benchmark code. The previously generated surrogate metrics live in the earlier separately delivered project. Do not mix its model results with actual new GRU–GNN evaluations.

## WQP–USGS field-data downloader (2019–2025)

The repository includes a **separate observational-data acquisition workflow**. It does not relabel the simulation-derived CSVs as observed data. It uses the **modern USGS daily values OGC API** (discharge 00060, mean statistic 00003), plus the **WQP WQX3 result download API** for discrete turbidity samples. An older WQP endpoint is a fallback for compatibility. The legacy USGS WaterServices API is not required.

```bash
python scripts/download_field_data.py --start 2019-01-01 --end 2025-12-31
python scripts/download_field_data.py --sites 01570500 01576000 --continue-on-error
```

The downloader writes source snapshots to `data/field_raw/` and the following auditable CSV outputs to `data/field_processed/`:

- `daily_discharge_all_sites.csv` (USGS daily discharge, cfs and m³/s)
- `discrete_turbidity_all_sites.csv` (WQP individual observations, units preserved)
- `matched_turbidity_discharge.csv` (exact station/date join, retaining multiple discrete samples)
- `daily_calendar_with_missingness.csv` (calendar and observed-sample indicator through missing cells)
- `download_manifest.json` (source URLs, counts, failed stations, provenance)

The demonstration station pair is Harrisburg `01570500` and Marietta `01576000`, Susquehanna River. The proposed link between them in `config/river_edge_candidates.csv` has **NOT** been verified using hydrological network tracing and **must not** be silently used as a directed neural edge. This pair does not form the seven-station graph required by the included simulated-data neural training code. No full-model real-field evaluation is claimed. WQP discrete results may be sparse, and measurement units such as NTU and FNU must not be pooled without defensible calibration.

**Optional API key:** Set `USGS_API_KEY` in your environment for higher rate limits. See `.env.example`. Do not commit API keys.

**API references:** https://api.waterdata.usgs.gov/docs/ogcapi/ , https://www.waterqualitydata.us/beta/webservices_documentation/

## GitHub upload

Create or use https://github.com/Qaisar256/HydroGraph-FL, extract the distributed ZIP archive, and push the extracted **contents** to the repository root. Run `python -m unittest discover -s tests -v` locally and inspect the git status before committing. The code contains training, evaluation, federated learning, QDE and source code for reproducible field data retrieval. GitHub publication does not imply any of the illustrative Table 7 neural model values are measured. The field downloader needs an internet connection and may depend on current availability of public services.

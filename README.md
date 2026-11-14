# HydroGraph-FL: Hydrology-Aware Federated Spatiotemporal Graph Learning for River Water Quality Forecasting

**HydroGraph-FL** is a Python-based research framework for river-water-quality forecasting that integrates Gated Recurrent Units (GRU), directed Graph Neural Networks (GNN), hydrology-aware learning, Federated Learning (FL), and Quantum-Inspired Differential Evolution (QDE). The framework combines historical water-quality measurements, streamflow observations, and upstream–downstream hydrological connectivity to support next-day turbidity forecasting and early-warning classification.

The project provides reproducible preprocessing, forecasting baselines, neural-model implementations, and federated optimization workflows for investigating water-quality prediction across distributed river-monitoring networks.

## Key Features

- **GRU Temporal Encoding:** Extracts temporal dependencies from historical turbidity, discharge, and related water-quality measurements.
- **Directed Graph Learning:** Supports upstream–downstream information propagation through hydrologically defined graph connections.
- **Hydrology-Aware Regularization:** Incorporates conditional transport-based constraints when sufficient physical observations are available.
- **Federated Learning:** Implements FedAvg and FedProx for collaborative model training across independent catchment clients.
- **QDE Hyperparameter Optimization:** Provides quantum-inspired differential evolution for neural-model hyperparameter selection.
- **Water-Quality Early Warning:** Supports next-day turbidity forecasting and training-derived threshold-based alert classification.
- **Reproducible Evaluation:** Includes chronological data partitioning, forecasting baselines, performance metrics, and model-training utilities.
- **Field-Data Integration:** Supports observational datasets and data retrieval from environmental monitoring services.

## Datasets

### 1. Ballycanew Field-Observation Dataset

The project includes observational water-quality data from the **Ballycanew agricultural catchment, County Wexford, Ireland**, originally reported by Negri et al. (2024).

The merged dataset covers **2009–2016** and contains:

| Dataset characteristic | Value |
|---|---:|
| Daily calendar records | 2,922 |
| Available turbidity observations | 2,574 |
| Available discharge observations | 2,649 |
| Available TRP observations | 2,649 |
| Valid one-day-ahead forecasting samples | 2,568 |

The forecasting data are partitioned chronologically:

| Partition | Period | Samples | Percentage |
|---|---|---:|---:|
| Training | 2009–2014 | 1,838 | 71.57% |
| Validation | 2015 | 365 | 14.21% |
| Testing | 2016 | 365 | 14.21% |

The dataset supports temporal forecasting, discharge-dependent analysis, and high-turbidity classification. Because it represents one monitored catchment, it does not independently establish multi-client federated learning or directed multi-station graph performance.

**Original dataset:** https://github.com/CamillaNegri/Ballycanew_Ptool

**Reference:** Negri C, Mellander P-E, Schurch N, Wade AJ, Gagkas Z, Wardell-Johnson DH, Adams K, Glendell M (2024). Bayesian network modelling of phosphorus pollution in agricultural catchments with high-resolution data. *Environmental Modelling & Software*, 178, 106073. https://doi.org/10.1016/j.envsoft.2024.106073

### 2. Reproducible River-Network Benchmark

A separate computational benchmark supports controlled evaluation of temporal predictors, upstream features, and federated linear optimization.

| Benchmark characteristic | Value |
|---|---:|
| Catchments | 5 |
| Monitoring stations | 35 |
| Daily observations per station | 540 |
| Station-day records | 18,900 |
| Held-out forecasting observations | 2,940 |
| Random seed | 20261109 |

The benchmark is simulation-based and is evaluated independently of the Ballycanew field observations. Results from the benchmark must not be reported as measured field-data performance.

### 3. Additional Environmental Data Sources

The project includes utilities for accessing additional river-water-quality and hydrological observations.

- Water Quality Portal: https://www.waterqualitydata.us/
- USGS Water Data: https://waterdata.usgs.gov/

These services can support future multi-catchment validation after monitoring stations, discharge gauges, measurement units, and hydrological connectivity have been verified.

## Installation

**Requirements:** Python 3.10 or later.

```bash
git clone https://github.com/Qaisar256/HydroGraph-FL.git
cd HydroGraph-FL
pip install -r requirements.txt
```

## Running HydroGraph-FL

### Neural Architecture and Federated Training

Run the complete set of supported neural experimentation modes:

```bash
python run_neural.py --mode all
```

Run individual training configurations:

```bash
# Federated Averaging
python run_neural.py --mode fedavg

# Federated Proximal Optimization
python run_neural.py --mode fedprox

# QDE-based hyperparameter optimization
python run_neural.py --mode qde
```

Run a shortened functionality test:

```bash
python run_neural.py --mode all --quick
```

These commands support architectural experimentation. Successful execution does not imply that the complete model has achieved the performance reported for separate baseline estimators.

### Download Environmental Observations

```bash
python scripts/download_field_data.py \
  --start 2019-01-01 \
  --end 2025-12-31
```

### Automated Tests

```bash
python -m unittest discover -s tests -v
```

## Model Evaluation

The framework supports regression and classification evaluation through:

- Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE).
- Classification accuracy, precision, recall, and F1-score.
- Receiver Operating Characteristic Area Under the Curve (ROC-AUC).
- Chronological validation and forward-chaining cross-validation.
- Client-level forecasting performance.
- Discharge-dependent prediction-error analysis.
- Federated communication-round monitoring.

## Experimental Results

The research repository distinguishes the following experimental settings:

**Computational benchmark:** Ridge regression and FedAvg-linear estimators were evaluated on common held-out benchmark observations. FedAvg Graph Linear achieved an MAE of 1.194 NTU and RMSE of 1.501 NTU. Graph-aware Ridge reduced MAE by 3.13% relative to temporal Ridge.

**Ballycanew field-observation evaluation:** A GRU-based temporal classification prototype achieved 84.11% mean accuracy under ten-fold forward-chaining validation and 82.19% accuracy on the independent 2016 test set.

**Proposed complete HydroGraph-FL architecture:** The integrated GRU–GNN, hydrology-aware regularization, FedAvg/FedProx, and QDE components are provided for research experimentation. The baseline results above must not be interpreted as experimentally verified performance of the complete architecture.

## Research Applications

HydroGraph-FL supports research in river-water-quality forecasting, streamflow-aware prediction, directed hydrological graph learning, distributed environmental monitoring, high-turbidity detection, and federated model optimization.

## Authors

**Alanoud Alsalem and Qaisar Abbas**

## Code and Data Availability

Project repository:

https://github.com/Qaisar256/HydroGraph-FL

Original Ballycanew observations:

https://github.com/CamillaNegri/Ballycanew_Ptool

## Research Status

HydroGraph-FL is an experimental research framework. Its Python implementation supports GRU–GNN modeling, FedAvg/FedProx optimization, QDE hyperparameter search, and baseline evaluation. Full multi-catchment field validation, calibrated hydrological transport constraints, and security-aware federated deployment remain subjects of further research.

## License

See the repository's `LICENSE` file for the applicable software license. Original third-party datasets remain subject to their respective source licenses and attribution requirements.

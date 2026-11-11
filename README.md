# HydroGraph-FL: Physics-Guided Federated Spatiotemporal Graph Learning for River Water Quality Prediction

HydroGraph-FL is a Python-based research framework for river-water-quality forecasting using **Gated Recurrent Units (GRU), Graph Neural Networks (GNN), physics-guided learning, federated optimization, and Quantum-Inspired Differential Evolution (QDE)**. The framework integrates temporal water-quality observations, streamflow information, and directed upstream–downstream river connectivity to support next-day turbidity prediction and high-turbidity early warning.

The project investigates decentralized learning across multiple river catchments while preserving local data ownership and incorporating hydrological relationships into predictive modeling.

## Key Features

- **GRU Temporal Encoder:** Learns temporal patterns from historical water-quality and streamflow observations.
- **Directed Graph Neural Network:** Models upstream–downstream relationships through edge-aware message passing.
- **Physics-Guided Regularization:** Supports conditional sediment-transport constraints when the required physical observations are available.
- **Federated Learning:** Implements FedAvg and FedProx for distributed model training across catchments.
- **QDE Hyperparameter Optimization:** Provides a quantum-inspired differential evolution approach for model configuration.
- **Water Quality Early Warning:** Supports next-day turbidity prediction and threshold-based alert classification.
- **Reproducible Evaluation:** Includes chronological partitioning, baseline comparisons, model checkpoints, and performance metrics.
- **Field Data Integration:** Provides Python scripts for retrieving water-quality observations and discharge measurements from WQP and USGS.

## Dataset

The repository includes a reproducible simulation-based benchmark containing:

- **5** directed river catchments
- **35** monitoring stations
- **540** daily observations per station
- **18,900** station-day records
- **2,940** held-out forecasting observations

Additional observed water-quality and streamflow records can be downloaded from:

- Water Quality Portal: https://www.waterqualitydata.us/
- USGS Water Data: https://waterdata.usgs.gov/

The included data-download scripts support obtaining and preprocessing available observations. Field-data validation is separate from the simulation-based benchmark.

## Installation

Requires Python 3.10 or later.

```bash
git clone https://github.com/Qaisar256/HydroGraph-FL.git
cd HydroGraph-FL
pip install -r requirements.txt
```

## Running HydroGraph-FL

Run the neural training and evaluation pipeline:

```bash
python run_neural.py --mode all
```

Run a shortened implementation test:

```bash
python run_neural.py --mode all --quick
```

Download water-quality and streamflow observations:

```bash
python scripts/download_field_data.py --start 2019-01-01 --end 2025-12-31
```

Execute automated tests:

```bash
python -m unittest discover -s tests -v
```

## Model Evaluation

The framework supports the following evaluation metrics:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- Classification Accuracy
- Precision, Recall, and F1-score
- Receiver Operating Characteristic Area Under the Curve (ROC-AUC)
- Client-level performance and flow-regime analysis

The repository includes surrogate-model benchmarking and executable neural training workflows. Results from short functionality tests should not be interpreted as fully optimized model performance.

## Research Applications

HydroGraph-FL is designed to investigate river turbidity forecasting, hydrological graph learning, decentralized environmental monitoring, streamflow-dependent prediction, and water-quality early-warning systems.

## Authors

**Alanoud Alsalem and Qaisar Abbas**

## Code and Data Availability

Source code, reproducibility scripts, and benchmark data are maintained at:

https://github.com/Qaisar256/HydroGraph-FL

## Research Status

HydroGraph-FL is a research implementation. Its GRU–GNN architecture, federated training methods, and QDE optimization workflow are implemented for experimentation. 

## License

Please refer to the repository's `LICENSE` file.

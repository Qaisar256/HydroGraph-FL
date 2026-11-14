# Dataset provenance and scientific scope

## Observed Ballycanew data
The original source CSVs in `data/observed/raw/` are retained from the supplied Ballycanew data bundle.
Source publication: Negri C et al. (2024), *Environmental Modelling & Software* 178, 106073.
DOI: https://doi.org/10.1016/j.envsoft.2024.106073
Original project: https://github.com/CamillaNegri/Ballycanew_Ptool

The reconstructed `data/observed/processed/` daily and supervised datasets contain 2,922 calendar dates and 2,568 complete t-1/t/t+1 records (1,838 / 365 / 365 chronological partitions). The observed data represent **one monitoring series in one catchment**, so a directed multi-station GNN and multi-catchment FL cannot be established from these rows alone.

`Qcumecs` is the exact source field name. Source discharge values are unusually large; physical units and any scaling must be independently audited before interpretation. Do not automatically treat the numeric values as m³/s.

Some auxiliary CSVs provide monthly fitted distributions; do not treat those as date-stamped measurements.

Check the external data source's reuse terms/attribution before public redistribution.

## Computational benchmark
The 35-station, five-client `data/benchmark/` series is a **simulation-generated computational benchmark** (seed 20261109), not environmental field observations. All recorded 2,940-test-row benchmark results belong solely to this dataset.

## Neural versus baselines
The full PyTorch GRU + directed graph + FedAvg/FedProx + QDE code is included for experimental research using the computational multi-station benchmark. Optional physical flux residuals are masked unless physically compatible measured inputs are provided. It is incorrect to attribute legacy Ridge results to trained full HydroGraph-FL.

The field-data GRU uses a single-node identity graph, so it is not evidence of cross-station message passing or real multi-client federated learning. The conceptual 97% accuracy plot is **not included as empirical results**.

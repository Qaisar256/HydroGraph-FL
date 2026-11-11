# HydroGraph-FL implementation overview

The included training pipeline is a **research prototype on a simulated five-catchment dataset**.

```text
14-day station histories (7 nodes per catchment, 4 features)
      │
      ▼
Gated recurrent unit (GRU) station encoder
      │  last hidden state per station
      ▼
Directed graph message-passing layers
      │  adjacency[downstream, upstream] = 1
      │  discharge-conditioned gated messages
      ▼
Shared neural regression readout
      │
      ▼
Next-day turbidity per station (NTU)
      │
      ├─ Central training with training-only scaling
      ├─ Federated training: FedAvg / FedProx, one client per catchment
      ├─ Optional verified masked physical flux residual
      └─ QDE search over hyperparameters using validation error
```

## Scientific boundaries

- River topology in the simulated CSV is known by construction. Proposed real-site candidate edges are unverified.
- The physics residual requires measured discharge, upstream sediment flux, source/sink and storage change, and validated conversion between turbidity (NTU) and sediment concentration (mg/L). With absent inputs, the masked physical loss has zero valid observations.
- No quantum computer is used. QDE is a classical quantum-inspired heuristic.
- Parameter exchange does **not** imply encryption, differential privacy, or secure aggregation.
- The supplied 2019–2025 downloader acquires observed records for a separate field-data pipeline; no complete real-field neural benchmark is claimed.
- Never cite illustrative model performance values as empirical experiments.

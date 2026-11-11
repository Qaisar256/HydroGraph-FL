# Data dictionary and provenance

## Included training data (`data/`)
- `generated_station_daily_observations.csv`: simulation-produced station-days, original signals (turbidity NTU, streamflow m³/s, rainfall mm, temperature °C).
- `forecast_supervised_rows.csv`: one-day-ahead supervised observations and chronological partitions.
- `directed_river_edges.csv`: 30 synthetic directed edges, from upstream sender to downstream receiver.
- `observed_physics_inputs_TEMPLATE.csv`: schema for optional independent physical terms, not measured evidence.

Original data creation seed: `20261109`. Five simulated catchments, seven stations each, 540 days per station. These CSVs must not be described as WQP–USGS observations.

## Downloaded observational records (`data/field_processed/`, after user runs downloader)
- `daily_discharge_all_sites.csv`: actual USGS daily-mean flow with cfs and m³/s, USGS site IDs, status and observation dates.
- `discrete_turbidity_all_sites.csv`: actual WQP discrete turbidity with unit, observation date, activity ID and source.
- `matched_turbidity_discharge.csv`: exact station-day matches, not interpolated.
- `daily_calendar_with_missingness.csv`: date grid, missing measurements preserved as null.
- `download_manifest.json`: URLs and counts; each run represents the retrieved snapshot.

**Do not assume** FNU and NTU have interchangeable calibration, discrete samples constitute continuous daily observations, or candidate edges prove reach-based routing.

# Federated Adaptive Forecasting for Heterogeneous IoT Sensor Networks Under Concept Drift

**Companion code** for the research paper *Federated Adaptive Forecasting for Heterogeneous IoT Sensor Networks Under Concept Drift*, by Lamir Shkurti et al. (2026, under review).

This repository contains a simulation framework for adaptive federated learning in heterogeneous IoT sensor networks. The framework is implemented as a standalone Python application, without Docker or Flask, and uses a single-process execution model.

The central idea is to enable each simulated client to make independent, round-by-round decisions based on its current device state and local data conditions. These decisions determine whether local training should take place, how many local epochs should be executed, when model updates should be transmitted, whether updates should be compressed, whether the local model should remain active, and whether the client should switch to a lightweight model variant.

## 1. Project Overview

This project introduces a new simulation framework built from the ground up to investigate adaptive federated learning under device heterogeneity, non-IID data distributions, and concept drift.

The implementation builds on the feature engineering and evaluation methodology established in our previous temperature forecasting research, including lag-based, rolling-window, and cyclical features, chronological data splitting, and skill-score evaluation against a persistence baseline.

The framework extends this methodology with seven additional capabilities:

- Battery and energy consumption modelling.
- RAM availability as an input to client-side decision-making.
- Model update compression.
- Dynamic model activation and deactivation.
- Switching between full and lightweight model variants.
- Non-IID data distribution across clients.
- Concept drift injection and detection.

These mechanisms are integrated into a rule-based adaptive controller that coordinates client behaviour throughout the federated learning process.

## 2. Installation

Clone the repository and install the required dependencies:

```bash
pip install -r requirements.txt
```

## 3. Dataset

**Refer to `data/README.md` for detailed dataset preparation instructions.**

The experiments use the **Environmental Sensor Telemetry Data** dataset available on Kaggle:

`garystafford/environmental-sensor-data-132k`

Download the dataset and place the CSV file at:

```text
data/iot_telemetry_data.csv
```

The dataset was selected for the following reasons:

- **Multiple physical devices:** The dataset contains measurements from three Raspberry Pi devices equipped with DHT22 and MQ sensor arrays. Device MAC addresses provide a natural basis for defining separate federated clients.
- **Non-IID characteristics:** Measurements collected by different devices reflect different physical conditions and locations, providing a basis for studying heterogeneous local data distributions without artificially partitioning a single data stream.
- **Forecasting target:** The `temp` variable is used as the temperature prediction target, maintaining consistency with our previous forecasting research.
- **Accessible data format:** The dataset is distributed as a CSV file and does not require API authentication.

## 4. Running Experiments

The framework supports two principal experimental configurations: a non-adaptive baseline and the proposed adaptive policy.

### 4.1. Non-adaptive baseline

The baseline follows a fixed execution strategy: clients train in every round and transmit full model updates.

```bash
python run_experiment.py --data data/iot_telemetry_data.csv --policy non_adaptive --rounds 80
```

### 4.2. Adaptive policy

The adaptive configuration enables the proposed client-side decision-making mechanisms.

```bash
python run_experiment.py --data data/iot_telemetry_data.csv --policy adaptive --rounds 80
```

### 4.3. Experiments with concept drift

Concept drift can be injected into a specific client to evaluate the response of the adaptive controller under changing data conditions.

```bash
python run_experiment.py --data data/iot_telemetry_data.csv --policy adaptive --rounds 80 \
    --drift "b8:27:eb,gradual_sensor_bias,40,3.0"
```

Each experiment generates two output files:

- `outputs/<run-name>.csv` — round-by-round, client-level experimental logs.
- `outputs/<run-name>_config.json` — complete experiment configuration and parameters, supporting reproducibility.

### 4.4. Result analysis and visualization

The `analyze_results.py` script compares experimental runs and generates summary statistics and visualizations.

```bash
python analyze_results.py \
    --runs outputs/adaptive_80r.csv outputs/non_adaptive_80r.csv \
    --labels Adaptive NonAdaptive \
    --output-dir outputs/figures
```

The analysis produces:

- `summary_table.csv` — aggregate performance and communication statistics, including mean RMSE, total transmitted bytes, and the percentage of rounds involving training, transmission, compression, lightweight model operation, and deactivation.
- `rmse_over_rounds.png` — RMSE evolution across communication rounds.
- `bytes_cumulative.png` — cumulative communication overhead.
- `decisions_<label>.png` — visualization of client-side decisions for each experimental configuration.

### 4.5. Client-specific concept drift analysis

The standard analysis script aggregates RMSE across all three clients. This is appropriate for evaluating the overall performance of the baseline configurations (Section 4.1 of the paper), but it can obscure the effects of concept drift when only one client is affected.

For this reason, `plot_drift_comparison.py` provides a separate analysis that tracks the RMSE of the targeted client.

```bash
python plot_drift_comparison.py \
    --abrupt-nonadaptive outputs/nonadaptive_80r_drift_abrupt.csv \
    --abrupt-adaptive outputs/adaptive_80r_drift_abrupt.csv \
    --gradual-nonadaptive outputs/nonadaptive_80r_drift_gradual.csv \
    --gradual-adaptive outputs/adaptive_80r_drift_gradual.csv \
    --output-dir outputs/figures
```

The default values for `--target`, `--start-round`, and `--ramp-rounds` correspond to the drift configuration used in the experiments. They can be overridden when different scenarios are required.

The script generates:

- `drift_rmse_comparison.png` — a two-panel comparison corresponding to Figure 4 in the paper.
- `drift_phase_summary.csv` — mean RMSE values for the pre-drift, drift, and post-drift phases, providing the data reported in Table 4.

An additional consistency check reports the maximum RMSE difference between the compared runs before the configured drift onset. This value should be zero. A non-zero result indicates that the experimental configurations differ in some respect, for example, due to different random seeds.

## 5. Adaptive Controller Architecture

The central component of the proposed framework is the adaptive controller implemented in `src/controller.py`.

It maps **nine input signals to six client-side decisions**.

### 5.1. Input signals

The controller considers the following signals:

1. Battery charge level.
2. CPU utilization.
3. Available RAM.
4. Network bandwidth.
5. Network latency.
6. Amount of newly available local data.
7. Deviation from the historical local data distribution.
8. Recent RMSE trend.
9. Concept drift detection ratio, based on an EWMA-to-baseline comparison.

### 5.2. Adaptive decisions

Based on these signals, each client independently determines:

1. **Training participation:** Whether to perform local training or skip the current round.
2. **Local training intensity:** Number of local epochs.
3. **Update transmission:** Whether to transmit the current update or retain it locally.
4. **Update representation:** Whether to send a full-precision or compressed update.
5. **Model availability:** Whether the local model remains active or becomes inactive.
6. **Model complexity:** Whether to use the full or lightweight model variant.

The decision thresholds are configurable through `ControllerThresholds` in `src/config.py`.

This configuration allows alternative threshold settings to be evaluated without modifying the controller implementation. Such experiments also support sensitivity analysis, which is particularly relevant when assessing the behaviour and robustness of a rule-based adaptive control mechanism.

## 6. Concept Drift Modelling

Concept drift injection and detection are implemented in `src/drift.py`.

The selected Kaggle dataset covers approximately one week of measurements and therefore does not provide sufficient temporal coverage for investigating long-term seasonal changes comparable to those observed in the Tetovo deployment data.

Consequently, concept drift is introduced explicitly through controlled simulation scenarios. These scenarios should be interpreted as injected drift rather than naturally occurring drift observed in the original dataset.

Two drift mechanisms are supported.

### 6.1. Gradual sensor bias

`gradual_sensor_bias` introduces an additive bias that increases progressively until the specified magnitude is reached within the configured number of rounds.

The bias is applied consistently to the temperature target and the corresponding temperature-related features. This scenario represents a gradual sensor offset, analogous to the self-heating behaviour observed in node4/UB.

An important methodological consideration is that, once the bias stabilizes, an autoregressive linear model can partially absorb the offset because the same shift is reflected in both the lag features and the target variable.

As a result, the measurable increase in forecasting error may be concentrated during the transition period rather than persisting throughout the entire post-drift phase. This distinction is relevant when interpreting the difference between transient drift effects and persistent changes in the underlying data relationships.

### 6.2. Abrupt relationship shift

`abrupt_relationship_shift` introduces a sudden change in the relationship between humidity and temperature.

Unlike the additive bias scenario, the altered relationship persists after the drift onset. This provides a suitable setting for evaluating the longer-term adaptation and recovery behaviour of the federated learning system.

The scenario follows the general concept drift taxonomy discussed by Gama et al. (2014).

### 6.3. Command-line configuration

The drift argument follows this format:

```text
--drift "<device_id_prefix>,<drift_type>,<start_round>,<magnitude>"
```

For example:

```bash
--drift "b8:27:eb,gradual_sensor_bias,40,3.0"
```

The device identifier is specified using a prefix because the original MAC addresses contain colon characters.

### 6.4. Important correction: drift round indexing (October 2026)

A correction was introduced in October 2026 to address an indexing issue in the concept drift injection mechanism.

**Experiments involving `--drift` conducted before this correction must be repeated. Their previous drift-related results should not be considered valid.**

Experiments without drift injection, including the adaptive and non-adaptive baselines and ablation experiments, are not affected.

The original implementation used the raw DataFrame index and treated each raw observation as an individual simulation round. However, a single simulation round typically contains multiple resampled observations.

With the default five-minute resampling frequency, approximately six to eight raw observations may correspond to one simulation round. Consequently, a configured drift onset at round 40 could effectively occur around round 5 or 6, while the gradual transition could finish substantially earlier than intended.

The issue was identified through the analysis of experimental logs, where the targeted client's RMSE increased considerably before the configured drift onset.

The implementation was corrected by calculating the chronological round boundaries using `_chronological_round_slices` before applying the drift transformation. This ensures that raw observations are mapped to their corresponding simulation rounds.

A regression test, `test_drift_never_touches_rounds_before_start_round`, was added to `tests/test_pipeline.py`. It verifies that all rounds preceding the configured drift onset remain identical between the no-drift and drift-enabled scenarios.

This test specifically addresses the temporal correctness of drift injection rather than merely checking whether drift eventually increases the prediction error.

## 7. Energy and Battery Model

The device energy model is implemented in `src/device_state.py`.

Battery state evolves throughout the simulation rather than being assigned independently at each communication round. Battery depletion depends on estimated current consumption, round duration, and the operating mode of the client.

The model distinguishes between:

- Idle energy consumption.
- Additional consumption during local training.
- Communication-related energy costs.

The device capacities and current consumption values are engineering assumptions rather than direct physical measurements. They are informed by published Raspberry Pi 4/5 power consumption figures, typical 10,000–20,000 mAh power-bank capacities, and the computational profile reported in our previous Arduino MKR1000 deployment study.

In particular, the previous study reported approximately 204–259 CPU cycles per prediction, corresponding to 4.3–5.4 μs at 48 MHz.

These values provide a reference for computational cost modelling. However, the simulated battery consumption should not be interpreted as experimentally measured energy consumption from physical IoT devices.

This distinction is important when interpreting the results and reporting the experimental methodology.

## 8. Automated Tests

The repository includes regression tests for the main simulation components.

Run the test suite using:

```bash
python tests/test_pipeline.py
```

Alternatively, if `pytest` is installed:

```bash
python -m pytest tests/ -v
```

The test suite covers:

1. Successful dataset loading and identification of the three devices.
2. Execution of the adaptive policy without NaN or Inf values.
3. Expected training and transmission behaviour of the non-adaptive baseline.
4. Error increase under gradual concept drift.
5. Correct drift onset, ensuring that earlier rounds remain unaffected.
6. Validation of excessive round counts relative to the available live samples.
7. Communication overhead comparison between adaptive and non-adaptive configurations.
8. Ablation testing of the update transmission mechanism.
9. Ablation testing of update compression.

The tests use a small synthetic dataset located at:

```text
tests/fixtures/tiny_iot_telemetry.csv
```

This fixture is intended exclusively for automated testing and continuous integration. It is not used to produce the experimental results reported in the paper.

## 9. Experimental Configuration and Reproducibility

The following steps are required to reproduce the main experiments.

### 9.1. Dataset preparation

Download the Kaggle dataset and place it at:

```text
data/iot_telemetry_data.csv
```

Refer to `data/README.md` for the complete instructions.

### 9.2. Baseline and adaptive experiments

Execute both configurations using the same dataset and consistent experimental settings.

A range of 60–120 rounds may be considered, depending on the resampling frequency and the amount of available live data.

Before increasing the number of rounds, however, it is important to consider the number of new observations available to each client per round.

### 9.3. Important: Minimum training samples and high round counts

A limitation related to the interaction between the number of rounds and the resampling frequency was identified and corrected in October 2026.

The total number of live observations, corresponding to the configured test fraction, remains fixed for a given resampling frequency. Increasing the number of simulation rounds therefore divides the same number of observations into smaller round-level batches.

With the default `5min` resampling frequency, the Kaggle dataset provides approximately 577 live observations per client.

For example:

- With 80 rounds, each client receives approximately 7 observations per round, which is sufficient for the default `min_new_samples_to_train=6` threshold.
- With 200 rounds, the number decreases to approximately 2.9 observations per round, falling below the minimum required for local training.

In the latter case, the adaptive controller may skip local training in every round. The experiment can still produce RMSE values because predictions are generated using the existing model parameters, but the resulting logs would not represent an actual training and communication process.

To prevent this situation, `prepare_clients()` now validates the configuration for the adaptive policy and raises a `ValueError` when the selected number of rounds would prevent meaningful training, unless the corresponding training-skip ablation is enabled.

For experiments requiring a larger number of rounds, increasing the temporal resolution through a smaller resampling interval is preferable to simply increasing `--rounds`.

The following 200-round configuration was tested successfully:

```bash
python run_experiment.py --data data/iot_telemetry_data.csv --policy adaptive \
    --rounds 200 --resample-freq 2min --round-minutes 14.41 \
    --run-name adaptive_200r

python run_experiment.py --data data/iot_telemetry_data.csv --policy non_adaptive \
    --rounds 200 --resample-freq 2min --round-minutes 14.41 \
    --run-name nonadaptive_200r
```

The `--round-minutes 14.41` parameter represents the approximate real-world duration associated with one simulation round under this configuration. It is used by the battery model and should remain consistent with the selected resampling frequency and number of rounds.

This configuration also enables the battery-constrained clients to reach a depleted battery state, allowing the model deactivation mechanism to be evaluated over an extended simulation.

### 9.4. Ablation experiments

Ablation experiments are supported directly through command-line arguments. Each flag disables one adaptive mechanism while leaving the remaining mechanisms enabled.

The available options are:

- `--ablate-train-skip`
- `--ablate-epoch-adapt`
- `--ablate-send-hold`
- `--ablate-compression`
- `--ablate-deactivation`
- `--ablate-lite-switch`

For example, to evaluate the contribution of update compression:

```bash
python run_experiment.py --data data/iot_telemetry_data.csv \
    --policy adaptive --rounds 80 \
    --ablate-compression \
    --run-name adaptive_80r_ablate_compression
```

The resulting experiment can be compared against the standard adaptive configuration:

```text
outputs/adaptive_80r.csv
outputs/adaptive_80r_ablate_compression.csv
```

The same procedure can be applied to the other adaptive mechanisms.

This experimental design makes it possible to isolate the contribution of individual controller dimensions while preserving the behaviour of the remaining adaptive components.

### 9.5. Result generation

After completing the experiments, run the analysis scripts to generate the summary tables and figures.

The resulting CSV files and visualizations provide the basis for evaluating predictive performance, communication overhead, client decisions, and adaptation under concept drift.

## 10. Repository Structure

```text
adaptive-fl-concept-drift/
│
├── src/
│   ├── config.py
│   ├── data_loader.py
│   ├── features.py
│   ├── models.py
│   ├── device_state.py
│   ├── drift.py
│   ├── compression.py
│   ├── controller.py
│   └── fl_simulation.py
│
├── tests/
│   ├── fixtures/
│   │   └── tiny_iot_telemetry.csv
│   └── test_pipeline.py
│
├── data/
│   └── README.md
│
├── outputs/
│
├── run_experiment.py
├── analyze_results.py
├── plot_drift_comparison.py
├── requirements.txt
├── LICENSE
└── README.md
```

### Main implementation components

| Component | Description |
|---|---|
| `src/config.py` | Experiment presets, device profiles, configuration parameters, and controller thresholds. |
| `src/data_loader.py` | Dataset loading, preprocessing, and device-specific data partitioning. |
| `src/features.py` | Lag-based, rolling-window, and cyclical feature engineering for full and lightweight model variants. |
| `src/models.py` | Linear regression model with gradient descent and gradient clipping, compatible with FedAvg aggregation. |
| `src/device_state.py` | Simulation of battery, CPU, RAM, and network state, including battery evolution. |
| `src/drift.py` | Concept drift injection and EWMA-based drift detection. |
| `src/compression.py` | 8-bit quantization and delta-based model update compression. |
| `src/controller.py` | Core adaptive controller implementing the mapping from nine input signals to six client-side decisions. |
| `src/fl_simulation.py` | Round-by-round orchestration of the federated learning simulation. |
| `run_experiment.py` | Command-line interface for experiment execution. |
| `analyze_results.py` | Aggregate result analysis, comparisons, summary tables, and visualizations. |
| `plot_drift_comparison.py` | Target-client-specific concept drift analysis and visualization. |

## 11. Citation

If you use this implementation in academic research, please cite the associated paper:

```bibtex
@article{shkurti2026fedadaptive,
  title   = {Federated Adaptive Forecasting for Heterogeneous IoT Sensor Networks Under Concept Drift},
  author  = {Shkurti, Lamir and TODO: Co-Author},
  journal = {TODO: Journal name},
  year    = {2026}
}
```

The BibTeX entry should be updated with the final author list, journal information, and DOI once the paper has been accepted and published.

## 12. License and Dataset Availability

The source code is released under the **MIT License**. See the `LICENSE` file for details.

The dataset is not included in this repository. The `data/iot_telemetry_data.csv` file must be obtained separately from Kaggle, as described in `data/README.md`.

The dataset remains subject to the licensing and usage conditions established by Kaggle and its original authors. The MIT License covering this repository does not extend to the third-party dataset.

---

**Research paper:** *Federated Adaptive Forecasting for Heterogeneous IoT Sensor Networks Under Concept Drift* (2026, under review).

**Author:** Lamir Shkurti et al.

**Repository:** `adaptive-fl-concept-drift`

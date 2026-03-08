# Battery Simulation & Data Pipeline Take-Home Assessment

**Time Estimate**: 4-6 hours (can be split across sessions)

## Important Notes:

- Completing this take-home assignment is not strictly required to be considered as a candidate.
- This take-home assignment is designed to be completed in your own time. You are not required to complete it within a specific time frame. 
- You are not required to complete all phases. The most important thing is to demonstrate your thought process and how you would approach the problem.
- If you wish to present your work which is related but not exactly the same as the take-home assignment, that is also acceptable as long as you can apply it uniquely to the data provided in the take-home assignment.
- At minimum phase 6 should be completed, or some slides should be prepared prior to a technical interview.
- You are free to use any tools or libraries you wish, including AI tools. You are free to do different phases in different programming languages if you wish. If you have an alternative to PyBaMM for battery simulation, that is also acceptable as long as it can be reproduced by the interviewer.


## Overview

You've received raw battery cycling data from a lab. Your task is to build a complete data pipeline that cleans, simulates, stores, and visualizes battery performance data.

## Getting Started

Download this repo. Make a private repository on Github and push this repo to it. Share it with the username `mleot`, and commit & push your work as you go.

**Do not fork this repository. Create a separate private repository.**

Use uv or pip to install requirements.txt into an environment of your choice. Install other dependencies as you see fit.

## Data

The file `data/raw_cycling_data.csv` contains cycling data for multiple battery cells undergoing unique cycling protocols. The data has typical lab data quality issues, and poor protocol design issues that you'll need to address.

---

## Phase 1: Data Ingestion & Cleaning

**Time estimate**: 15-45 minutes  
**Deliverable**: Code for task & csv output 

### Your task:

1. Load the raw cycling data from `data/raw_cycling_data.csv`
2. Identify and handle missing values
3. Ensure consistency and data availability in relevant columns (voltage, current, test time)
4. Correct cycle/step indexing to be sequential
5. Document your cleaning decisions

### Phase 1 implementation in this repo

`phase1_cleaning.py` performs per-cell cleaning for `data/raw_cycling_data.csv` and writes a cleaned dataset plus before/after plots.

- Input: `data/raw_cycling_data.csv` (uses `datetime`, `cell_id`, `cycle`, `step`, `current_A`, `voltage_V`)
- Processing scope: each `cell_id` is cleaned independently, then concatenated back
- Datetime quality checks:
  - parses `datetime` and reports missing/invalid timestamp count
  - checks strict increment within each `cell_id`
  - drops rows with non-incremental timestamps (`datetime <= previous datetime` in the same cell)
- Missing-value interpolation (`current_A`, `voltage_V`):
  - if both adjacent rows are same `step` and non-missing -> average
  - elif previous adjacent row is same `step` and non-missing -> copy previous
  - elif next adjacent row is same `step` and non-missing -> copy next
  - else -> keep missing
- Interpolation audit: prints decision counts (`if`, `elif_prev`, `elif_next`, `none`) per signal and per cell
- Step correction (phase 1 indexing):
  - first, fixes short anomalous runs (`<5` rows) when surrounding runs have identical `step` values
  - then, reindexes step chunks sequentially from `1..N` within each cycle for all cells
- Cycle correction: if cycle jumps by more than +1, normalizes the tail so transitions are at most +1 while preserving continuity
- Outputs (per cell): raw and cleaned timeseries plots for `current_A`, `voltage_V`, `cycle`, `step`
- Outputs (global): `data/phase1_cleaned_data.csv`

#### Run Phase 1 cleaning

```bash
python phase1_cleaning.py
```

Key output files:

- `data/phase1_cleaned_data.csv`
- `outputs/phase1/01_raw_timeseries_CELL_A.png` (and other cells)
- `outputs/phase1/02_cleaned_timeseries_CELL_A.png` (and other cells)

---

## Phase 2: Simulation & Comparison

**Time estimate**: 45-60 minutes  
**Deliverable**: Code for simulated cycling data using PyBaMM

### Your task:

1. Use PyBaMM to simulate identical cycling profiles as the experimental data.
2. Choose an appropriate model (DFN, SPMe, SPM, etc.) and appropriate submodels and justify your choices.
3. Configure the simulation to match the experimental conditions. You may need to make assumptions or simplifications.
4. Run the simulations and compare simulated vs experimental voltage curves
5. Document your process, problems solved, and general findings

**Resources**:
- [PyBaMM Documentation](https://docs.pybamm.org/)
- [PyBaMM Models](https://docs.pybamm.org/en/stable/source/user_guide/models/)

### Phase 2 implementation in this repo

`phase2_simulation.py` replays the cleaned experimental current profile (`I(t)`) for each selected cell using the selected model (`SPM` / `SPMe` / `DFN`) and compares simulated terminal voltage against measured voltage.

- Input: `data/phase1_cleaned_data.csv` (uses `datetime`, `cell_id`, `cycle`, `step`, `current_A`, `voltage_V`)
- Model selection: `SPM`, `SPMe`, or `DFN` (`--model-name`, default `SPMe`)
- Parameter set: selectable PyBaMM set name via `--parameter-set` (for example, `Chen2020` or `Chen2020_composite`)
- Capacity handling: `capacity_ah` is applied to nominal capacity and used to scale parallel-electrode count so capacity impacts current-driven dynamics
- Reusable API for Phase 4:
  - `make_comparison_df(...)` builds per-cell simulation-vs-experiment DataFrame
  - `simulate_cells(...)` returns `(summary_df, comparison_by_cell)` and can skip file writes with `save_files=False`
- Summary metadata includes model/config fields (`model_name`, `solver_mode`, `voltage_min`, `voltage_max`) for DB insertion
- For composite parameter sets (for example, `Chen2020_composite`), the script enables composite model options automatically and, if `initial_soc` initialization is not supported by PyBaMM, falls back to parameter-set default initial state while recording `initial_soc_applied=false` in summary output.
- Outputs on successful completion:
  - per cell: `outputs/phase2/<CELL>_comparison.csv`, `outputs/phase2/<CELL>_voltage_compare.png`
  - global: `outputs/phase2/simulation_summary.csv` with RMSE by cell and run metadata

#### Single-cell run

```bash
python phase2_simulation.py \
  --cells CELL_A \
  --capacity-ah 5.0 \
  --initial-soc 0.95 \
  --model-name SPMe \
  --parameter-set Chen2020
```

#### Multi-cell run (per-cell capacity and SoC)

Create a JSON file, for example `data/phase2_cell_config.json` (storing values hand-tuned):

```json
{
  "CELL_A": { "capacity_ah": 6.88, "initial_soc": 0.85 },
  "CELL_B": { "capacity_ah": 1.125, "initial_soc": 0.60 },
  "CELL_C": { "capacity_ah": 5.6, "initial_soc": 0.91 },
  "CELL_D": { "capacity_ah": 1.2, "initial_soc": 0.77 },
  "CELL_E": { "capacity_ah": 0.204, "initial_soc": 0.80 }
}

```

Then run:

```bash
python phase2_simulation.py \
  --cells CELL_A CELL_B CELL_C CELL_D CELL_E \
  --cell-config-json data/phase2_cell_config.json \
  --model-name DFN \
  --parameter-set Chen2020
```

To override voltage limits, add `--voltage-max 4.5 --voltage-min 2.0` (these are defaults).
To simulate only early life cycles, add `--max-cycle <N>` (inclusive), e.g. `--max-cycle 50`.

Following codes are used for initial guesses before hand-tuning capacity and initial SoCs for each cell.

#### Capacity pre-estimation for Phase 2 inputs

Use `phase2_capacity_estimation.py` to estimate per-cell capacity from the requested windows:

- A: cycle 1, step 5 (CC discharge)
- B: cycle 1, step 7 (fast CC discharge) and cycle 2, step 2 (slow CC discharge)
- C: cycle 2, step 3 (CC discharge)
- D: cycle 2, step 1 and step 2 (CC + CV charge)
- E: cycle 2, step 5 (CC discharge)

Run:

```bash
python phase2_capacity_estimation.py
```

Outputs:

- `outputs/phase2/capacity_window_details.csv` (window-by-window integration)
- `outputs/phase2/capacity_estimates.csv` (per-cell estimated capacity)
- `outputs/phase2/phase2_cell_config_from_capacity.json` (ready for `--cell-config-json`)

#### Capacity decay from early vs late cycles

Use `phase2_capacity_decay.py` to compute capacity decay using the same step definitions:

- A: step 5
- B: step 7
- C: step 3
- D: steps 1 and 2
- E: step 5

Comparison windows:

- `CELL_A`, `CELL_B`, `CELL_C`, `CELL_E`: first cycle vs last cycle

Special baseline rule:

- `CELL_D`: baseline is fixed to cycle 2 (steps 1 and 2), then compared to the last cycle

Run:

```bash
python phase2_capacity_decay.py
```

Outputs:

- `outputs/phase2/capacity_decay_details.csv` (per cycle/step integration details)
- `outputs/phase2/capacity_decay_summary.csv` (per-cell decay in Ah and %)

#### Initial SoC estimation using first-step fitting

Use `phase2_initial_soc_estimation.py` to estimate initial SoC for each cell by:

- taking each cell's first step (first contiguous cycle/step block in time)
- replaying that step's current in SPMe
- iterating initial SoC to match first-step voltage end-point
- using bisection when bracketed, with a grid fallback when not bracketed

Example:

```bash
python phase2_initial_soc_estimation.py \
  --cells CELL_A CELL_B CELL_C CELL_D CELL_E \
  --capacity-estimates-csv outputs/phase2/capacity_estimates.csv \
  --parameter-set Chen2020
```

Outputs:

- `outputs/phase2/initial_soc_estimates.csv`
- `outputs/phase2/phase2_cell_config_with_estimated_soc.json` (capacity + estimated initial SoC)

---

## Phase 3: Database Design

**Time estimate**: 30-45 minutes  
**Deliverable**: Database schema (SQL or diagram or other)

### Your task:

Design a database schema that logically supports these query patterns:

1. "Show all cycles for CELL_A"
2. "Compare simulation vs experiment for cycle 2"
3. "List all parameter sets used with the DFN model"
4. "Show voltage curves for all cells simulated using SPMe model with default parameters"

Consider how to store:
- Cell configurations
- Experimental data (cleaned)
- Model types and parameter sets
- Simulation results

You do not need to implement the database, just design the schema. You can use any database schema design tool or just describe the schema in text using exact key names and relationship maps.

### Phase 3 implementation in this repo

designed as a SQLite schema.

Tables:

- `cells`: cell master (`CELL_A`, etc.)
- `experimental_runs`: experiment metadata (`cell_id`, `start_time_ts_utc`, `end_time_ts_utc`, `profile`, `environment`)
- `experimental_timeseries`: cleaned experiment data (`test_time_s`, `cycle`, `step`, `current`, `voltage`)
- `parameter_sets`: base set name + `name_extention` + `modified_parameters_json` + `base_parameters_json`
- `simulation_runs`: simulation metadata per cell, parameter set, and model (`model_name`, `capacity_ah`, `initial_soc`)
- `simulation_timeseries`: simulated data (`test_time_s`, `cycle`, `step`, `current`, `voltage`)

Deliverables:

- `phase3_database_schema.sql` (DDL schema definition)

Relationship map:

- `cells (1) -> (N) experimental_runs`
- `experimental_runs (1) -> (1) experimental_timeseries`
- `cells (1) -> (N) simulation_runs`
- `parameter_sets (1) -> (N) simulation_runs`
- `simulation_runs (1) -> (1) simulation_timeseries`

How this supports the required queries:

1. all cycles for `CELL_A`: `cells` + `experimental_runs` + `experimental_timeseries`
2. compare simulation vs experiment for cycle 2: `experimental_runs` + `experimental_timeseries` + `simulation_runs` + `simulation_timeseries`
3. list parameter sets used with DFN: `simulation_runs` filtered by `model_name = 'DFN'` + `parameter_sets`
4. show SPMe default voltage curves for all cells: `simulation_runs` (`SPMe`) + `cells.default_parameter_set_id` + `parameter_sets` + `simulation_timeseries`

---

## Phase 4: Database Population & Visualization

**Time estimate**: 45-60 minutes  
**Deliverable**: Populated database, visualization script, and plot output example images

### Your task:

1. Create the database (SQLite recommended)
2. Populate with cleaned experimental data
3. Run simulations for multiple parameter variations and models from phase 2
4. Store simulation results
5. Create a **decoupled** visualization script that:
   - Reads from the database
   - Plots experimental vs simulated curves
   - Can be run independently

### Phase 4 database population implementation in this repo

`phase4_database_population.py` populates the SQLite database defined by `phase3_database_schema.sql`.

- Experimental source: `data/phase1_cleaned_data.csv` (cleaned Phase 1 output)
- Simulation source/config:
  - current profile replay from cleaned data
  - per-cell `capacity_ah` and `initial_soc` from `data/phase2_cell_config.json`
  - supports multiple models in one run (default: `SPM` and `SPMe`)
- Population modes:
  - `full`: insert experimental + simulation
  - `experimental-only`: insert only cleaned experimental data
  - `simulation-only`: insert only simulation data (useful for incremental appends later, e.g., optimized parameter sets)

#### Full population (cleaned experiment + SPM/SPMe simulation)

```bash
python phase4_database_population.py \
  --db-path outputs/phase4/battery_pipeline.db \
  --schema-sql-path phase3_database_schema.sql \
  --cleaned-csv-path data/phase1_cleaned_data.csv \
  --cell-config-json data/phase2_cell_config.json \
  --mode full \
  --models SPM SPMe \
  --parameter-set Chen2020 \
  --recreate-db
```

#### Simulation-only append (additional runs later)

Example for appending optimized-parameter simulation runs without touching experimental rows:

```bash
python phase4_database_population.py \
  --db-path outputs/phase4/battery_pipeline.db \
  --cleaned-csv-path data/phase1_cleaned_data.csv \
  --cell-config-json data/phase2_cell_config.json \
  --mode simulation-only \
  --models SPM SPMe \
  --parameter-set Chen2020 \
  --parameter-name-extention "_optimized_v1" \
  --modified-parameters-json data/optimized_params_v1.json \
  --run-name optimized_v1
```

Use `--replace-existing-simulation` only when you want to overwrite simulation runs matching `(cell, model, parameter_set, name_extention, run_name)`.
`simulation_runs` enforces uniqueness for `(cell, model, parameter_set, run_name)` when `run_name` is not null, which avoids ambiguous base-run lookup while preserving multi-cell batch run labels.
When `--parameter-name-extention ""` (default parameter set), `cells.default_parameter_set_id` is assigned to the resolved default parameter-set row for each selected cell.

### Phase 4 visualization implementation in this repo

`phase4_plot_from_db.py` is the DB-backed visualization/query script.
- Optional cycle filtering via `--cycle` (single cycle or consecutive range, e.g., `1-2`)
- Optional stacked current subplot via `--plot-with-current`
- With `--plot-with-current`, the current subplot includes a secondary C-rate axis
  computed as current divided by `simulation_runs.capacity_ah`
- Supports plotting multiple simulation runs and multiple cells in one figure
- When multiple cells are selected, each cell is drawn in separate subplot panels
- Supports multi-value filters via `--models` and `--parameter-sets`
- By default, omitted filters mean "all" (all cells/models/parameter sets/run names)
- Curve selection mode via `--series-mode {both,experiment-only,simulation-only}`
- Parameter-set listing mode via `--list-parameters`
- Optimization-run listing mode via `--list-optimization-runs`
- Default parameter-set inspection via `--show-default-parameter-set`

#### Minimal usage (one simulation run id)

```bash
python phase4_plot_from_db.py \
  --db-path outputs/phase4/battery_pipeline.db \
  --simulation-run-ids 1
```

#### Plot selected cells with multiple models in one figure

```bash
python phase4_plot_from_db.py \
  --db-path outputs/phase4/battery_pipeline.db \
  --cells CELL_A CELL_B \
  --models SPM SPMe \
  --parameter-sets Chen2020
```

#### Plot explicit simulation run ids together

```bash
python phase4_plot_from_db.py \
  --db-path outputs/phase4/battery_pipeline.db \
  --simulation-run-ids 1 3 8
```

#### Plot cycle range (1-2) with current

```bash
python phase4_plot_from_db.py \
  --db-path outputs/phase4/battery_pipeline.db \
  --cells CELL_A CELL_B \
  --models SPMe \
  --parameter-sets Chen2020 \
  --cycle 1-2 \
  --plot-with-current
```

#### Plot SPMe + default parameters for all cells

Default parameter sets are cell-specific via `cells.default_parameter_set_id`.

```bash
python phase4_plot_from_db.py \
  --db-path outputs/phase4/battery_pipeline.db \
  --models SPMe \
  --default-parameters-only
```

Output files are written to `outputs/phase4/` by default.

### Commands for the 4 required query patterns

Use `phase4_plot_from_db.py` as the DB-backed consumer for each pattern below.

#### 1) "Show all cycles for CELL_A"

```bash
python phase4_plot_from_db.py \
  --db-path outputs/phase4/battery_pipeline.db \
  --cells CELL_A \
  --series-mode experiment-only
```

#### 2) "Compare simulation vs experiment for cycle 2"

```bash
python phase4_plot_from_db.py \
  --db-path outputs/phase4/battery_pipeline.db \
  --cycle 2
```

Optionally scope to one cell/model:

```bash
python phase4_plot_from_db.py \
  --db-path outputs/phase4/battery_pipeline.db \
  --cells CELL_A \
  --models SPMe \
  --cycle 2
```

#### 3) "List all parameter sets used with the DFN model"

Use this command to explicitly list DFN parameter sets:

```bash
python phase4_plot_from_db.py \
  --db-path outputs/phase4/battery_pipeline.db \
  --models DFN \
  --list-parameters
```

#### List optimization runs and settings

```bash
python phase4_plot_from_db.py \
  --db-path outputs/phase4/battery_pipeline.db \
  --cells CELL_B \
  --models SPM \
  --list-optimization-runs
```

#### 4) "Show voltage curves for all cells simulated using SPMe model with default parameters"

Default parameter sets are cell-specific via `cells.default_parameter_set_id`.

```bash
python phase4_plot_from_db.py \
  --db-path outputs/phase4/battery_pipeline.db \
  --models SPMe \
  --default-parameters-only \
  --show-default-parameter-set
```

---

## Phase 5: Parameter Optimization

**Time estimate**: 45-60 minutes  
**Deliverable**: Optimized parameter set

### Your task:

1. Define an objective function (e.g., RMSE between simulated and experimental voltage)
2. Choose parameters to optimize (e.g., diffusivity, conductivity)
3. Implement optimization using scipy.optimize or similar
4. Document the optimization approach and results
5. Store optimized parameters in the database

Note: You do not need to simulate every possible combination, nor achieve a "perfect" fit. The goal is to demonstrate your approach to parameter optimization.

### Phase 5 implementation in this repo

Phase 5 is split into:

- `phase5_sensitivity.py`: sensitivity analysis and plotting
- `phase5_run_optimization.py`: optimization run
- `phase5_evaluate_optimization_runs.py`: DB-backed comparison across optimization runs

- Supports cycle filtering with `--cycle` (`2` or `1-2`)
- Replays measured current profile using Phase 2 simulation logic
- Objective uses weighted sum of category-specific RMSE windows (default `1:1:1:1`):
  - ohmic: `0-2 s` from each step start
  - kinetic: `2-20 s`
  - diffusion: `20-120 s`
  - capacity: `>=120 s`
- PyBaMM is always solved on the full experimental time grid during optimization (stage 1 and stage 2).
- Tail downsampling is applied only when computing the capacity window RMSE (`>=120 s`), per step segment:
  - `--tail-downsample-stride-stage1`: stage-1 capacity-window RMSE stride
  - `--tail-downsample-stride-stage2`: stage-2/final capacity-window RMSE stride
- Weights are configurable via `--rmse-weights OHMIC KINETIC DIFFUSION CAPACITY`
- Primary optimizer is metaheuristic `differential_evolution` (`scipy.optimize`)
- Optimization variables (max five):
  - initial SoC (`--optimize-initial-soc`; bounds via `--initial-soc-scale-bounds`)
  - capacity (`--optimize-capacity`; bounds via `--capacity-scale-bounds`)
  - one ohmic selector (`--ohmic-parameter {n,p,e}`)
  - one kinetic selector (`--kinetic-parameter {n,p}`; `e` is not valid)
  - one diffusion selector (`--diffusion-parameter {n,p,e}`)
- Candidate families are restricted to:
  - diffusion: `Negative/Positive particle diffusivity`, `Electrolyte diffusivity`
  - reaction: `Negative/Positive electrode exchange-current density`
  - ohmic: `Negative/Positive electrode conductivity`, `Electrolyte conductivity`
- Function-valued parameters (e.g., `Electrolyte diffusivity`, `Electrolyte conductivity`, exchange-current densities in `Chen2020`) are scaled by wrapping the original function output.
- Stage-1 now optimizes all selected variables by default (`--stage1-variable-limit` is deprecated and ignored, kept only for CLI compatibility).
- Each optimization run now automatically:
  - exports DB-ready artifacts (`*_db_modified_parameters.json`, `*_optimized_cell_config.json`)
  - appends the optimized simulation result to DB (`phase4_database_population.py`, simulation-only mode)
  - calls DB-backed comparison plotting (`phase4_plot_from_db.py`) with current subplot enabled by default
- Optimization metadata is persisted in DB table `optimization_runs` (linked to `simulation_runs.id`) for traceability.
- If `--db-parameter-name-extention` collides with an existing parameter-set row that has different overrides, a collision-safe suffix is appended automatically (old rows are preserved).
- You can start optimization from a previous DB run via `--base-simulation-run-id` or `--base-run-name` (optionally add `--base-run-parameter-name-extention` only when run names are reused and you want explicit disambiguation).
- `simulation_runs` enforces uniqueness for `(cell, model, parameter_set, run_name)` when `run_name` is not null; this keeps base-run lookup deterministic while allowing the same run name across different cells or parameter sets.
- Sensitivity uses simulation-vs-simulation deltas (baseline simulated voltage vs perturbed simulated voltage), so experimental voltage is not required for sensitivity scoring.
- Sensitivity parameter perturbations default to `x0.5, x0.8, x1.0, x1.2, x1.5, x2.0` (that is: `-50%`, `-20%`, baseline, `+20%`, `+50%`, `+100%`).
- Sensitivity no longer sweeps capacity or initial SoC; those are fixed per run.
- `phase5_sensitivity.py` can read capacity/SoC from `data/phase2_cell_config.json` via `--cell-config-json` when `--capacity-ah`/`--initial-soc` are omitted.

#### Run sensitivity analysis first (recommended)

```bash
python phase5_sensitivity.py \
  --cell CELL_A \
  --cycle 1-2 \
  --model-name SPM \
  --parameter-set Chen2020 \
  --cell-config-json data/phase2_cell_config.json \
  --sensitivity-include-all-candidates \
  --rmse-weights 1 1 1 1
```

Outputs:

- `outputs/phase5/CELL_A_sensitivity_details.csv`
- `outputs/phase5/CELL_A_sensitivity_ranked.csv`
- `outputs/phase5/CELL_A_sensitivity_plots/*.png`

#### Run optimization on selected parameters (fresh baseline)

```bash
python phase5_run_optimization.py \
  --cell CELL_A \
  --cycle 1-2 \
  --capacity-ah 6.88 \
  --initial-soc 0.85 \
  --model-name SPM \
  --parameter-set Chen2020 \
  --optimize-initial-soc \
  --optimize-capacity \
  --initial-soc-scale-bounds 0.9 1.1 \
  --capacity-scale-bounds 0.8 1.2 \
  --ohmic-scale-bounds 0.5 2.0 \
  --kinetic-scale-bounds 0.2 5.0 \
  --diffusion-scale-bounds 0.2 5.0 \
  --rmse-weights 1 1 1 1 \
  --ohmic-parameter n \
  --kinetic-parameter n \
  --diffusion-parameter n \
  --stage1-maxiter 8 \
  --stage1-popsize 6 \
  --stage1-seed 42 \
  --local-method Powell \
  --stage2-maxiter 40 \
  --db-path outputs/phase4/battery_pipeline.db \
  --db-parameter-name-extention "_phase5_optimized" \
  --db-run-name "opt_cell_a_spm_v1"
```

#### Run optimization using a previous optimized run as base

First, list candidate runs and their metadata:

```bash
python phase4_plot_from_db.py \
  --db-path outputs/phase4/battery_pipeline.db \
  --cells CELL_A \
  --models SPM \
  --list-optimization-runs
```

Then warm-start from a previous run name:

```bash
python phase5_run_optimization.py \
  --cell CELL_A \
  --cycle 1-2 \
  --model-name SPM \
  --parameter-set Chen2020 \
  --optimize-initial-soc \
  --optimize-capacity \
  --ohmic-parameter n \
  --kinetic-parameter n \
  --diffusion-parameter n \
  --rmse-weights 1 1 1 1 \
  --stage1-maxiter 8 \
  --stage1-popsize 6 \
  --stage1-seed 42 \
  --local-method Powell \
  --stage2-maxiter 40 \
  --db-path outputs/phase4/battery_pipeline.db \
  --db-parameter-name-extention "_phase5_optimized" \
  --db-run-name "opt_cell_a_spm_v2" \
  --base-run-name "opt_cell_a_spm_v1"
```

If run names are reused and you need explicit disambiguation, add:
`--base-run-parameter-name-extention "<name_extention>"` or use `--base-simulation-run-id`.

Outputs:

- `outputs/phase5/CELL_A_optimization_stage1_history.csv`
- `outputs/phase5/CELL_A_optimization_stage2_history.csv`
- `outputs/phase5/CELL_A_best_result.json`
- `outputs/phase5/CELL_A_best_voltage_trace_full.csv`
- `outputs/phase5/CELL_A_db_modified_parameters.json`
- `outputs/phase5/CELL_A_optimized_cell_config.json`
- `outputs/phase5/phase5_optimization_comparison.csv` (from run-comparison utility)
- Appended simulation run in `outputs/phase4/battery_pipeline.db` (`run_name`: UTC auto-generated unless `--db-run-name` is set)
- DB-backed comparison plot in `outputs/phase4/` (with current subplot by default)
- Traceability row in `optimization_runs` for the appended simulation run (`simulation_run_id`, `base_simulation_run_id`, objective, and full optimization config JSON)

#### Compare optimization runs (DB-backed)

```bash
python phase5_evaluate_optimization_runs.py \
  --db-path outputs/phase4/battery_pipeline.db \
  --cells CELL_A \
  --models SPM \
  --parameter-set Chen2020 \
  --latest 10 \
  --sort-by objective
```

This prints a comparison table with:

- final parameters (`capacity_ah`, `initial_soc`, optimized variable values, parameter scales)
- voltage RMSE metrics (`0-2s`, `2-20s`, `20-120s`, `120+s`, and full-profile)

CSV output is saved by default to:

- `outputs/phase5/phase5_optimization_comparison.csv`

---

## Phase 6: Results Preparation

**Time estimate**: 30-45 minutes  
**Deliverable**: ≤6 slide presentation

### Your presentation should cover:

1. **Data Challenges**: What issues did you find and how did you handle them?
2. **Architecture Decisions**: How did you structure your code and database?
3. **Database Design**: How did you design your database schema and why is it effective for this problem?
4. **Optimization Results**: Which parameters improved fit? By how much?
5. **AI Usage**: How did you use AI tools (if any)? What worked well? What needed correction?
6. **Scaling Considerations**: How would this scale to 1000+ cells?

---

## Deliverables Checklist

- [ ] Cleaned data (CSV and in database)
- [ ] Database file (SQLite, populated)
- [ ] All source code (well-documented, comments)
- [ ] requirements.txt or similar
- [ ] Visualization example outputs (png format, or other image format, or markdownfiles with images embedded, or PDFs)
- [ ] Presentation (PDF, pptx, or other format, ≤6 slides)

---

## Evaluation Criteria

You'll be assessed on:

- **Code Quality**: Readable, maintainable, well-documented
- **Data Handling**: Appropriate cleaning decisions
- **Architecture**: Database design, code organization
- **Model Understanding**: Appropriate model selection and configuration
- **Optimization**: Sensible approach to parameter fitting
- **Communication**: Clear presentation of decisions and results

Good luck!


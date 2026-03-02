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
- Datetime check: parses `datetime` and reports missing/invalid timestamp count
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

`phase2_simulation.py` replays the cleaned experimental current profile (`I(t)`) for each selected cell using an SPMe model and compares simulated terminal voltage against measured voltage.

- Input: `data/phase1_cleaned_data.csv` (uses `datetime`, `cell_id`, `current_A`, `voltage_V`)
- Model: SPMe, no degradation submodels enabled by default
- Parameter set selection: `Chen2020` or `Chen2020_composite`
- Composite support: optional silicon ratio (`--si-ratio`) for `Chen2020_composite`
- Outputs (per cell): comparison csv + voltage overlay plot
- Outputs (global): `outputs/phase2/simulation_summary.csv` with RMSE by cell

#### Single-cell run

```bash
python phase2_simulation.py \
  --cells CELL_A \
  --capacity-ah 5.0 \
  --initial-soc 0.95 \
  --parameter-set Chen2020
```

#### Multi-cell run (per-cell capacity and SoC)

Create a JSON file, for example `data/phase2_cell_config.json`:

```json
{
  "CELL_A": { "capacity_ah": 5.0, "initial_soc": 0.95 },
  "CELL_B": { "capacity_ah": 4.9, "initial_soc": 0.93 },
  "CELL_C": { "capacity_ah": 5.1, "initial_soc": 0.90 }
}
```

Then run:

```bash
python phase2_simulation.py \
  --cells CELL_A CELL_B CELL_C \
  --cell-config-json data/phase2_cell_config.json \
  --parameter-set Chen2020_composite \
  --si-ratio 0.08
```

If current-sign convention appears inverted relative to PyBaMM in your environment, add `--current-sign -1.0`.

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

Composite example:

```bash
python phase2_initial_soc_estimation.py \
  --cells CELL_A CELL_B CELL_C CELL_D CELL_E \
  --capacity-estimates-csv outputs/phase2/capacity_estimates.csv \
  --parameter-set Chen2020_composite \
  --si-ratio 0.08
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


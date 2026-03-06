-- Phase 3 schema (SQLite).

PRAGMA foreign_keys = ON;

CREATE TABLE cells (
    id INTEGER PRIMARY KEY,
    cell_code TEXT NOT NULL UNIQUE,      -- e.g., CELL_A
    default_parameter_set_id INTEGER,
    FOREIGN KEY (default_parameter_set_id) REFERENCES parameter_sets(id) ON DELETE SET NULL
);

CREATE TABLE experimental_runs (
    id INTEGER PRIMARY KEY,
    cell_id INTEGER NOT NULL,
    start_time_ts_utc TEXT NOT NULL,
    end_time_ts_utc TEXT NOT NULL,
    profile TEXT,               -- e.g., "CCCV cycling at 1C"
    environment TEXT,                    -- e.g., "temperature chamber 25C"
    FOREIGN KEY (cell_id) REFERENCES cells(id) ON DELETE CASCADE
);

CREATE TABLE experimental_timeseries (
    id INTEGER PRIMARY KEY,
    experimental_run_id INTEGER NOT NULL,
    test_time_s REAL NOT NULL,
    cycle_index INTEGER NOT NULL,
    step_index INTEGER NOT NULL,
    current_a REAL,
    voltage_v REAL,
    temperature_c REAL,
    FOREIGN KEY (experimental_run_id) REFERENCES experimental_runs(id) ON DELETE CASCADE
);

CREATE INDEX idx_exp_run_cycle_step
    ON experimental_timeseries (experimental_run_id, cycle_index, step_index);

CREATE TABLE parameter_sets (
    id INTEGER PRIMARY KEY,
    base_parameter_set_name TEXT NOT NULL,  -- e.g., Chen2020
    name_extention TEXT NOT NULL DEFAULT '', -- e.g., "_tuned_v1"
    modified_parameters_json TEXT NOT NULL DEFAULT '{}',
    UNIQUE (base_parameter_set_name, name_extention)
);

CREATE TABLE simulation_runs (
    id INTEGER PRIMARY KEY,
    cell_id INTEGER NOT NULL,
    parameter_set_id INTEGER NOT NULL,
    model_name TEXT NOT NULL,               -- SPM, SPMe, DFN
    capacity_ah REAL,                    -- run-specific config
    initial_soc REAL,                    -- run-specific config
    run_name TEXT,                       -- optional label
    FOREIGN KEY (cell_id) REFERENCES cells(id) ON DELETE CASCADE,
    FOREIGN KEY (parameter_set_id) REFERENCES parameter_sets(id) ON DELETE RESTRICT
);

CREATE TABLE simulation_timeseries (
    id INTEGER PRIMARY KEY,
    simulation_run_id INTEGER NOT NULL,
    test_time_s REAL NOT NULL,
    cycle_index INTEGER,
    step_index INTEGER,
    current_a REAL,
    voltage_v REAL,
    temperature_c REAL,
    FOREIGN KEY (simulation_run_id) REFERENCES simulation_runs(id) ON DELETE CASCADE
);

CREATE INDEX idx_sim_run_cycle_step
    ON simulation_timeseries (simulation_run_id, cycle_index, step_index);
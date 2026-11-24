# Handoff: correction memory (record + replay manual fixes) — 2026-11-24 22:21

## Objective and status

The user wants the repetitive manual correction of badly ML-fitted spectra automated.
Agreed design (confirmed with the user before implementing): **no trained model** — record
the user's manual correction steps in an opt-in sidecar log, then *replay* the nearest
logged fixes onto badly fitted spectra. Applying proposals requires **one explicit
confirmation per batch** (repo rule: predictions never alter spectra without an explicit
user action); every applied change is undoable (Ctrl+Z) and logged as `applied_by: "replay"`.

**Status: complete, committed and pushed to `main` as `93eb99e` ("Add correction memory…").
Full test suite green (221 tests). Headless GUI smoke run passed.**

## What was done (commit 93eb99e)

- `ml/user_actions.py` — new sidecar `eis-fitting-user-actions` v1, written atomically to
  `<project>_user_actions.json` next to the `.eisfit.json[.gz]` (`user_actions_path_for`).
  Entries keyed by path-independent `ml.results_schema.spectrum_identifier`; no raw arrays
  duplicated. Contains structured payload diffs (`diff_state_payloads` / `diff_cycle_dict` /
  `diff_parameters`: parameter initial/lower/upper/fixed, circuit, frequency window, point
  masks, fit presence), `make_action_entry`, `append_user_actions` (trimmed to
  `MAX_USER_ACTIONS=5000`), tolerant `load_user_actions*`, `correction_entries`, and
  `relative_fit_residual`.
- `ml/correction_replay.py` — GUI-independent. `SpectrumFitSnapshot`, configurable feature
  list `DEFAULT_SIMILARITY_FEATURES = ("time", "voltage", "current", "cycle", "cycle mod")`
  with alias table `FEATURE_ALIASES` (change these if the feature set changes later),
  `effective_features` (drops features missing for most spectra), `expected_parameters`
  (k-NN, range-normalized distance, inverse-square weights, geometric mean for positive
  params — mirrors `eis_services.suggest_eec_parameters`), `flag_bad_fits` (robust **local**
  z = (actual − median of neighbor transformed values) / (1.4826·MAD + relative floor),
  per parameter; flags `parameter_outlier` / `fit_failed` / `high_residual`),
  `propose_corrections` (nearest logged entries with structurally compatible circuit via
  `circuit_structure.parameter_name_mapping`; replays parameter + equivalent-circuit
  changes only — never point masks or frequency windows).
- `eis_services.py` — added public alias `anomaly_parameter_transform = _anomaly_parameter_transform`.
- `eis_gui.py` —
  - imports for the two new modules;
  - `__init__`: `_record_user_actions=False`, `_correction_z_threshold="4"`,
    `_replay_in_progress`, `_pending_replay_sources`, `_last_correction_proposals`;
  - `_load_preferences` reads `correction_memory` (`record_user_actions`,
    `bad_fit_z_threshold`); `_save_preferences` gained keyword-only `correction_memory`
    (defaults to current attrs so the other caller keeps working) and writes the key;
  - Preferences → ML pipelines tab: "Correction memory" LabelFrame with the record
    Checkbutton and z-threshold Entry; validated on save;
  - recording hook: `_run_history_checkpoint` now calls `_maybe_record_user_actions`
    (builds entries from `diff_state_payloads`, residual before from the previous history
    snapshot, after from the live cycle, `replay_sources` attached from
    `_pending_replay_sources`) and clears `_replay_in_progress`; `_baseline_history` clears
    replay state;
  - menu bar: **Suggest corrections…** → `suggest_recorded_corrections` (selection or whole
    dataset; flags computed over the full dataset context) → worker via `_submit` →
    `_finish_correction_suggestions` review dialog (table: spectrum/cycle/V/I/time/reasons/
    worst param z/residual/proposed changes/source) → **Apply selected and refit** /
    **Apply selected** → `askyesno` confirm → `_apply_correction_proposals` (mutates via
    `state.parameters_for` copies + `cycle.clear_fit()`, validates bounds, sets
    `_pending_replay_sources`) → `batch_fit_spectra` → `_finish_correction_fit` →
    `_finish_explorer_batch_fit` + `refine_fit_selected(rows)`.
- `docs/ml-processing.md` — new "Correction memory" section.
- Tests: `tests/test_user_actions.py`, `tests/test_user_actions_integration.py`,
  `tests/test_correction_replay.py` (31+ tests: round-trip, tolerant load, diff extraction,
  real-payload recording integration, planted-outlier flagging, cross-circuit mapping,
  nearest-source preference, no replay of masks/windows).

## Decisions made and why

- Sidecar instead of project fields: avoids a `PROJECT_VERSION` bump; log travels with the
  project folder but isn't scientific state. Recording is opt-in (privacy/control).
- Replay (not a learned model): user explicitly chose this; the repo already had k-NN
  machinery (`suggest_eec_parameters`) to mirror.
- Local robust z (median/MAD of nearest neighbors in transformed space), not a global MAD:
  a global MAD flags trend endpoints and lets one outlier mask its neighbors (found during
  testing). Transform reuses `anomaly_parameter_transform` (log10 for R/L/CPE_Q,
  logit for CPE alpha).
- Only parameter/circuit changes are replayed; point masks and frequency windows are
  spectrum-specific (repo rule: never force-match masks).
- `effective_features` drops unavailable features (e.g. datasets without a time column)
  so flagging still works there.

## Open questions / next steps (priority order)

1. **Use it for real**: enable recording in Preferences → ML pipelines, correct several bad
   spectra by hand on a real project (e.g. a `467_*` cathode project), save, then run
   **Suggest corrections…** and check whether proposals match what you would have done.
   Tune `bad_fit_z_threshold` (default 4) if it over/under-flags.
2. Consider whether replayed **fitted** values (post-refit) are better sources than the
   logged *initial* values — currently only the logged change targets are replayed.
3. Optional: extend `max_sources` in `propose_corrections` (currently 1) to blend several
   logged fixes; optional residual threshold for flagging (`residual_threshold` param
   exists in `flag_bad_fits` but is not yet exposed in the GUI).
4. If the similarity feature set needs to change: edit `DEFAULT_SIMILARITY_FEATURES` /
   `FEATURE_ALIASES` in `ml/correction_replay.py`.
5. Possible follow-up: surface replay proposals automatically after the ML pipeline
   finishes (currently a manual menu action).

## How to verify

- Tests: `./.venv/Scripts/python.exe -m unittest discover -s tests` (last run: 221 tests, OK).
- Focused: `./.venv/Scripts/python.exe -m unittest tests.test_user_actions tests.test_user_actions_integration tests.test_correction_replay -v`
- GUI: `python main.py <path-to-small>.mpt --cycle 1`, then Preferences → ML pipelines
  (toggle recording), edit a parameter, save, then menu **Suggest corrections…**.
- Handoff smoke test used in this session: instantiate `EISApplication` with a synthetic
  multi-cycle dataframe via `load_project_from_dataframe`, register with
  `_register_dataset`/`_switch_dataset`, set `project_path` + `_record_user_actions`, run
  `_run_history_checkpoint()`, then `suggest_recorded_corrections()` + pump `root.update()`.

## Uncommitted changes

None at handoff time (working tree clean after commit 93eb99e; `report.log` and
`__pycache__` are pre-existing untracked artifacts, intentionally not staged).

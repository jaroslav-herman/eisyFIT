"""MCP server exposing eisyFIT project operations to any MCP-capable agent.

Run with ``python -m mcp_server`` or the ``eisyfit-mcp`` console script; the
default (and only) transport is stdio, so the client spawns this process and
talks over stdin/stdout (see ``docs/mcp-server.md`` for the Vibe config).

The server is GUI-independent by design: it never imports ``eis_gui`` and
drives the same service layer (``eis_services``, ``eis_project``, ``ml.*``)
as the editor, so repository invariants hold here too — mutations that
invalidate a fit clear it, spectra are matched by ``spectrum_identifier()``,
saves are atomic, and point masks / frequency windows are never force-matched
across spectra.

Safety model: tools that change user data (``set_parameters``,
``set_point_mask``, ``set_frequency_window``, ``apply_outliers``,
``apply_corrections``) are dry-run by default and only mutate when called with
``confirm=true``; they return the exact diff either way.  Applied changes are
appended to the project's ``<project>_user_actions.json`` sidecar with
``applied_by: "mcp"`` so GUI-recorded and agent-applied corrections share one
audit trail.  ``save_project`` writes a ``.bak`` copy of the previous project
file before overwriting.
"""

from __future__ import annotations

import argparse
import json
import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from circuit_structure import circuits_equivalent
from eis_model import CycleState, ParameterValue, ProjectState
from eis_project import export_fit_parameters_for_states, save_project_file
from eis_services import (
    FittedEECRecord,
    FitOptions,
    LoadedProject,
    SpectrumFitTarget,
    _fit_provenance,
    analyze_outliers,
    batch_fit_spectra,
    calculate_hybrid_drt,
    calculate_lin_kk_residuals,
)
from ml.correction_replay import (
    DEFAULT_SIMILARITY_FEATURES,
    CorrectionProposal,
    SpectrumFitSnapshot,
    effective_features,
    flag_bad_fits as _flag_bad_fits_impl,
    propose_corrections as _propose_corrections_impl,
)
from ml.results_schema import spectrum_identifier
from ml.user_actions import (
    CycleChange,
    ParameterChange,
    append_user_actions,
    correction_entries,
    load_user_actions,
    make_action_entry,
    relative_fit_residual,
    user_actions_path_for,
)

log = logging.getLogger("eisyfit.mcp")

DEFAULT_CIRCUIT = "R0-L0-p(R1,CPE1)"
DEFAULT_CONTROL = "cell"
PROJECT_SUFFIXES = (".eisfit.json.gz", ".eisfit.json", ".json")
MAX_ARRAY_POINTS = 2000


@dataclass
class _OpenProject:
    path: Path
    stamp: tuple[int, int]
    datasets: list[tuple[str, LoadedProject, ProjectState]] = field(default_factory=list)

    @property
    def key(self) -> str:
        return str(self.path)


class _Session:
    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.projects: dict[str, _OpenProject] = {}


_SESSION = _Session()


def _stamp(path: Path) -> tuple[int, int]:
    stat = path.stat()
    return (stat.st_mtime_ns, stat.st_size)


def _load_datasets(path: Path) -> list[tuple[str, LoadedProject, ProjectState]]:
    from eis_services import load_projects, load_saved_project

    if path.name.endswith(PROJECT_SUFFIXES):
        return load_saved_project(path)
    report = load_projects([path], DEFAULT_CONTROL, DEFAULT_CIRCUIT, 1)
    if report.errors:
        raise ValueError(report.errors[0][1])
    if not report.loaded:
        raise ValueError(f"No impedance spectra could be loaded from {path.name}")
    return [(dataset_id, loaded, loaded.state) for dataset_id, loaded in report.loaded]


def _get_project(project: str) -> _OpenProject:
    """Return the open project for ``project``, auto-opening it on first use."""
    path = Path(str(project)).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"No such project or data file: {path}")
    key = str(path)
    with _SESSION.lock:
        opened = _SESSION.projects.get(key)
        if opened is None:
            opened = _OpenProject(path=path, stamp=_stamp(path), datasets=_load_datasets(path))
            _SESSION.projects[key] = opened
        return opened


def _reload(opened: _OpenProject) -> _OpenProject:
    with _SESSION.lock:
        fresh = _OpenProject(path=opened.path, stamp=_stamp(opened.path), datasets=_load_datasets(opened.path))
        _SESSION.projects[fresh.key] = fresh
        return fresh


def _ensure_cycle(loaded: LoadedProject, cycle_number: int) -> CycleState:
    state = loaded.state
    cycle = state.cycles.get(cycle_number)
    if cycle is None:
        from eis_services import load_cycle

        cycle = load_cycle(loaded.dataframe, cycle_number, state.control)
        if state.all_frequency_window is not None:
            cycle.frequency_window = state.all_frequency_window
        cycle.parameters = state.parameters_for(cycle_number)
        cycle.circuit = state.circuit
        state.cycles[cycle_number] = cycle
    return cycle


def _spectrum_of(loaded: LoadedProject, cycle_number: int):
    return next((s for s in loaded.spectra if s.cycle == cycle_number), None)


def _resolve(opened: _OpenProject, identifier: str):
    """Resolve ``identifier`` to ``(dataset_id, loaded, cycle, spectrum)``.

    Accepts a ``spectrum_identifier`` hash, a ``dataset_id::cycle-N`` snapshot
    identity, ``dataset_id::cycle``/``dataset_id::N``, or a bare cycle number
    (first dataset).
    """
    identifier = str(identifier).strip()
    datasets = opened.datasets
    if identifier.startswith("eisfit-spectrum-v1:"):
        for dataset_id, loaded, _state in datasets:
            for spectrum in loaded.spectra:
                cycle = _ensure_cycle(loaded, spectrum.cycle)
                computed = spectrum_identifier(
                    cycle.frequency_hz,
                    cycle.impedance.real,
                    cycle.impedance.imag,
                    spectrum.cycle,
                    loaded.state.control,
                )
                if computed == identifier:
                    return dataset_id, loaded, cycle, spectrum
        raise ValueError(f"No spectrum matches identifier {identifier!r}")
    if "::" in identifier:
        dataset_id, _, cycle_part = identifier.rpartition("::")
        cycle_text = cycle_part.removeprefix("cycle-").removeprefix("cycle")
        try:
            cycle_number = int(cycle_text)
        except ValueError:
            raise ValueError(
                f"Cannot parse cycle from {identifier!r}; use '<dataset_id>::cycle-<n>'"
            ) from None
        for candidate_id, loaded, _state in datasets:
            if candidate_id == dataset_id:
                cycle = _ensure_cycle(loaded, cycle_number)
                return dataset_id, loaded, cycle, _spectrum_of(loaded, cycle_number)
        raise ValueError(f"Unknown dataset_id {dataset_id!r} in {identifier!r}")
    try:
        cycle_number = int(identifier)
    except ValueError:
        raise ValueError(
            f"Unrecognized spectrum identifier {identifier!r}; use a spectrum id, "
            "'<dataset_id>::cycle-<n>', or a bare cycle number"
        ) from None
    dataset_id, loaded, _state = datasets[0]
    cycle = _ensure_cycle(loaded, cycle_number)
    return dataset_id, loaded, cycle, _spectrum_of(loaded, cycle_number)


def _spectrum_summary(dataset_id: str, loaded: LoadedProject, spectrum, cycle: CycleState) -> dict:
    fit_ok = (
        cycle.fit_parameters is not None
        and cycle.fit_at_data_impedance is not None
        and np.asarray(cycle.fit_parameters).size == len(cycle.parameters)
    )
    return {
        "identifier": spectrum_identifier(
            cycle.frequency_hz,
            cycle.impedance.real,
            cycle.impedance.imag,
            spectrum.cycle,
            loaded.state.control,
        ),
        "identity": f"{dataset_id}::cycle-{spectrum.cycle}",
        "dataset_id": dataset_id,
        "dataset_label": loaded.dataset_label,
        "cycle": int(spectrum.cycle),
        "control": loaded.state.control,
        "circuit": cycle.model(loaded.state.circuit),
        "point_count": int(spectrum.point_count),
        "active_points": int(np.count_nonzero(cycle.included)),
        "min_frequency_hz": float(spectrum.minimum_frequency_hz),
        "max_frequency_hz": float(spectrum.maximum_frequency_hz),
        "potential_v": float(spectrum.potential_v),
        "current_ma": float(spectrum.current_ma),
        "time_s": spectrum.time_s,
        "has_fit": bool(fit_ok),
        "residual": relative_fit_residual(
            cycle.frequency_hz, cycle.impedance, cycle.fit_at_data_impedance, cycle.included
        ),
    }


def _parameter_dict(parameter: ParameterValue) -> dict:
    return {
        "name": parameter.name,
        "unit": parameter.unit,
        "initial": float(parameter.initial),
        "lower": float(parameter.lower),
        "upper": float(parameter.upper),
        "fixed": bool(parameter.fixed),
        "error_percent": parameter.error_percent,
    }


def _store_fit(cycle: CycleState, parameters: list[ParameterValue], result) -> None:
    cycle.fit_parameters = np.asarray(result.fitted_parameters, dtype=float)
    cycle.fit_frequency_hz = result.fit_frequency_hz
    cycle.fit_impedance = result.fit_impedance
    cycle.fit_at_data_impedance = result.fit_at_data_impedance
    cycle.fit_provenance = _fit_provenance(result) if hasattr(result, "options") else dict(
        getattr(result, "fit_provenance", {}) or {}
    )
    for parameter, fitted, error in zip(parameters, result.fitted_parameters, result.errors_percent):
        parameter.initial = float(fitted)
        parameter.error_percent = float(error)
    cycle.parameters = parameters


def _snapshot(dataset_id: str, loaded: LoadedProject, spectrum) -> SpectrumFitSnapshot:
    cycle = _ensure_cycle(loaded, spectrum.cycle)
    metadata = dict(cycle.custom_metadata)
    metadata.update(spectrum.custom_metadata)
    metadata.setdefault("potential_v", cycle.potential_v)
    metadata.setdefault("current_ma", cycle.current_ma)
    time_s = cycle.time_s if cycle.time_s is not None else spectrum.time_s
    metadata.setdefault("time_s", time_s)
    metadata.setdefault("Time", time_s)
    metadata.setdefault("cycle", int(spectrum.cycle))
    metadata.setdefault("Cycle mod 15", int(spectrum.cycle) % 15)
    fit_ok = (
        cycle.fit_parameters is not None
        and cycle.fit_at_data_impedance is not None
        and np.asarray(cycle.fit_parameters).size == len(cycle.parameters)
    )
    return SpectrumFitSnapshot(
        spectrum_id=spectrum_identifier(
            cycle.frequency_hz,
            cycle.impedance.real,
            cycle.impedance.imag,
            int(spectrum.cycle),
            loaded.state.control,
        ),
        dataset_id=dataset_id,
        dataset_label=loaded.dataset_label,
        cycle=int(spectrum.cycle),
        control=loaded.state.control,
        circuit=cycle.model(loaded.state.circuit),
        parameter_names=tuple(parameter.name for parameter in cycle.parameters),
        fitted_parameters=(
            tuple(float(value) for value in cycle.fit_parameters) if fit_ok else None
        ),
        fit_ok=bool(fit_ok),
        residual=relative_fit_residual(
            cycle.frequency_hz, cycle.impedance, cycle.fit_at_data_impedance, cycle.included
        ),
        metadata=metadata,
    )


def _log_action(
    opened: _OpenProject,
    dataset_id: str,
    loaded: LoadedProject,
    cycle: CycleState,
    action: str,
    change: CycleChange,
    *,
    replay_sources=None,
) -> None:
    """Append one ``applied_by="mcp"`` entry to the user-actions sidecar."""
    try:
        spectrum = _spectrum_of(loaded, cycle.cycle)
        metadata = dict(cycle.custom_metadata)
        if spectrum is not None:
            metadata.update(spectrum.custom_metadata)
        metadata.setdefault("potential_v", cycle.potential_v)
        metadata.setdefault("current_ma", cycle.current_ma)
        time_s = cycle.time_s if cycle.time_s is not None else (spectrum.time_s if spectrum else None)
        metadata.setdefault("time_s", time_s)
        metadata.setdefault("Time", time_s)
        metadata.setdefault("cycle", cycle.cycle)
        metadata.setdefault("Cycle mod 15", cycle.cycle % 15)
        entry = make_action_entry(
            applied_by="mcp",
            action=action,
            dataset_id=dataset_id,
            dataset_label=loaded.dataset_label,
            control=loaded.state.control,
            spectrum_id=spectrum_identifier(
                cycle.frequency_hz,
                cycle.impedance.real,
                cycle.impedance.imag,
                cycle.cycle,
                loaded.state.control,
            ),
            cycle=cycle.cycle,
            circuit=cycle.model(loaded.state.circuit),
            metadata=metadata,
            change=change,
            residual_after=relative_fit_residual(
                cycle.frequency_hz, cycle.impedance, cycle.fit_at_data_impedance, cycle.included
            ),
        )
        if replay_sources:
            entry["replay_sources"] = [dict(source) for source in replay_sources]
        append_user_actions(
            user_actions_path_for(opened.path),
            [entry],
            source_project=str(opened.path),
            similarity_features=list(DEFAULT_SIMILARITY_FEATURES),
        )
    except (OSError, ValueError, TypeError, KeyError) as error:
        log.warning("user action recording failed: %s", error)


def _proposal_dict(proposal: CorrectionProposal) -> dict:
    return {
        "identity": proposal.identity,
        "spectrum_id": proposal.snapshot.spectrum_id,
        "dataset_id": proposal.snapshot.dataset_id,
        "cycle": proposal.snapshot.cycle,
        "reasons": list(proposal.flag.reasons),
        "max_abs_z": proposal.flag.max_abs_z,
        "residual": proposal.flag.residual,
        "distance": proposal.distance,
        "changes": [dict(change) for change in proposal.changes],
        "sources": [dict(source) for source in proposal.sources],
    }


# ---------------------------------------------------------------------------
# Tools: session / project lifecycle
# ---------------------------------------------------------------------------


def open_project(project: str) -> dict:
    """Open a .eisfit.json[.gz] project or a raw EIS data file (.mpt, .txt, ...).

    Returns the project key (use it as the ``project`` argument of every other
    tool) plus a per-dataset summary.  Re-opening an already open project
    reloads it from disk.
    """
    with _SESSION.lock:
        path = Path(str(project)).expanduser().resolve()
        opened = _SESSION.projects.get(str(path))
        if opened is None or opened.stamp != _stamp(path):
            opened = _reload(_OpenProject(path=path, stamp=_stamp(path), datasets=[]))
        return {
            "project": opened.key,
            "file": opened.path.name,
            "datasets": [
                {
                    "dataset_id": dataset_id,
                    "dataset_label": loaded.dataset_label,
                    "control": loaded.state.control,
                    "circuit": loaded.state.circuit,
                    "cycles": len(loaded.spectra),
                    "technique": loaded.technique,
                }
                for dataset_id, loaded, _state in opened.datasets
            ],
        }


def list_projects() -> dict:
    """List currently open projects with their dataset summaries."""
    with _SESSION.lock:
        return {
            "projects": [
                {
                    "project": opened.key,
                    "file": opened.path.name,
                    "datasets": [dataset_id for dataset_id, _l, _s in opened.datasets],
                }
                for opened in _SESSION.projects.values()
            ]
        }


def close_project(project: str) -> dict:
    """Close a project and drop its in-memory state (does not modify files)."""
    with _SESSION.lock:
        opened = _get_project(project)
        _SESSION.projects.pop(opened.key, None)
        return {"closed": opened.key}


def reload_project(project: str) -> dict:
    """Discard in-memory changes and reload the project from disk."""
    with _SESSION.lock:
        opened = _reload(_get_project(project))
        return {"project": opened.key, "datasets": [d for d, _l, _s in opened.datasets]}


# ---------------------------------------------------------------------------
# Tools: inspection
# ---------------------------------------------------------------------------


def list_spectra(project: str, dataset_id: str | None = None) -> dict:
    """List spectra of a project (optionally one dataset) with fit status."""
    with _SESSION.lock:
        opened = _get_project(project)
        spectra = []
        for ds_id, loaded, _state in opened.datasets:
            if dataset_id is not None and ds_id != dataset_id:
                continue
            for spectrum in loaded.spectra:
                cycle = _ensure_cycle(loaded, spectrum.cycle)
                spectra.append(_spectrum_summary(ds_id, loaded, spectrum, cycle))
        return {"project": opened.key, "count": len(spectra), "spectra": spectra}


def get_spectrum(
    project: str,
    identifier: str,
    include_arrays: bool = False,
    max_points: int = MAX_ARRAY_POINTS,
) -> dict:
    """Full detail for one spectrum: metadata, parameters, fit, optional arrays."""
    with _SESSION.lock:
        opened = _get_project(project)
        dataset_id, loaded, cycle, spectrum = _resolve(opened, identifier)
        summary = _spectrum_summary(dataset_id, loaded, spectrum, cycle)
        summary["frequency_window"] = (
            list(cycle.frequency_window) if cycle.frequency_window is not None else None
        )
        summary["parameters"] = [_parameter_dict(p) for p in cycle.parameters]
        if include_arrays:
            cap = max(0, int(max_points))
            frequency = np.asarray(cycle.frequency_hz, dtype=float)
            impedance = np.asarray(cycle.impedance, dtype=complex)
            included = np.asarray(cycle.included, dtype=bool)
            if cap and frequency.size > cap:
                step = int(np.ceil(frequency.size / cap))
                index = np.arange(0, frequency.size, step)
            else:
                index = np.arange(frequency.size)
            summary["arrays"] = {
                "note": "subsampled" if index.size < frequency.size else "complete",
                "index": [int(i) for i in index],
                "frequency_hz": [float(v) for v in frequency[index]],
                "z_real": [float(v) for v in impedance[index].real],
                "z_imag": [float(v) for v in impedance[index].imag],
                "included": [bool(v) for v in included[index]],
            }
        return summary


def get_parameters(project: str, identifier: str) -> dict:
    """Parameter table (initial/lower/upper/fixed/error) for one spectrum."""
    with _SESSION.lock:
        opened = _get_project(project)
        dataset_id, loaded, cycle, _spectrum = _resolve(opened, identifier)
        return {
            "identity": f"{dataset_id}::cycle-{cycle.cycle}",
            "circuit": cycle.model(loaded.state.circuit),
            "parameters": [_parameter_dict(p) for p in cycle.parameters],
        }


def get_fit(project: str, identifier: str) -> dict:
    """Fit result for one spectrum: fitted values, errors, rmse, provenance."""
    with _SESSION.lock:
        opened = _get_project(project)
        dataset_id, loaded, cycle, _spectrum = _resolve(opened, identifier)
        provenance = dict(cycle.fit_provenance or {})
        fitted = None
        if cycle.fit_parameters is not None and len(cycle.parameters):
            fitted = [
                {"name": p.name, "value": float(v), "error_percent": p.error_percent}
                for p, v in zip(cycle.parameters, np.asarray(cycle.fit_parameters, dtype=float))
            ]
        return {
            "identity": f"{dataset_id}::cycle-{cycle.cycle}",
            "circuit": cycle.model(loaded.state.circuit),
            "has_fit": fitted is not None,
            "converged": provenance.get("converged"),
            "objective": provenance.get("objective"),
            "rmse": provenance.get("rmse"),
            "elapsed_seconds": provenance.get("elapsed_seconds"),
            "pipeline": provenance.get("pipeline"),
            "residual": relative_fit_residual(
                cycle.frequency_hz, cycle.impedance, cycle.fit_at_data_impedance, cycle.included
            ),
            "fitted_parameters": fitted,
        }


def list_user_actions(project: str, limit: int = 100) -> dict:
    """Read the recorded user/agent corrections sidecar of a project."""
    with _SESSION.lock:
        opened = _get_project(project)
        entries = load_user_actions(user_actions_path_for(opened.path))
        trimmed = entries[-max(0, int(limit)):] if limit else entries
        return {
            "project": opened.key,
            "sidecar": str(user_actions_path_for(opened.path)),
            "total": len(entries),
            "corrections": len(correction_entries(entries)),
            "entries": trimmed,
        }


# ---------------------------------------------------------------------------
# Tools: computation (fit, DRT, outliers, suggestions)
# ---------------------------------------------------------------------------


def _fit_options(method: str) -> FitOptions:
    return FitOptions(method=str(method)).validated()


def fit_spectrum(
    project: str,
    identifier: str,
    circuit: str | None = None,
    timeout_seconds: float = 60.0,
    method: str = "least_squares",
) -> dict:
    """Fit one spectrum with its current parameters (optionally an equivalent circuit)."""
    from eis_services import fit_cycle_with_timeout

    with _SESSION.lock:
        opened = _get_project(project)
        dataset_id, loaded, cycle, _spectrum = _resolve(opened, identifier)
        target_circuit = cycle.model(loaded.state.circuit)
        if circuit is not None:
            if not circuits_equivalent(target_circuit, circuit):
                raise ValueError(
                    f"circuit {circuit!r} is not structurally equivalent to {target_circuit!r}; "
                    "change the model in the GUI first"
                )
            cycle.circuit = circuit
            target_circuit = circuit
        parameters = loaded.state.parameters_for(cycle.cycle)
        result = fit_cycle_with_timeout(
            cycle, target_circuit, parameters, float(timeout_seconds), _fit_options(method)
        )
        _store_fit(cycle, parameters, result)
        loaded.state.cycles[cycle.cycle] = cycle
        return {
            "identity": f"{dataset_id}::cycle-{cycle.cycle}",
            "circuit": target_circuit,
            "converged": bool(result.converged),
            "rmse": float(result.rmse),
            "objective": float(result.objective),
            "elapsed_seconds": float(result.elapsed_seconds),
            "residual": relative_fit_residual(
                cycle.frequency_hz, cycle.impedance, cycle.fit_at_data_impedance, cycle.included
            ),
            "fitted_parameters": [
                {"name": p.name, "value": float(v), "error_percent": float(e)}
                for p, v, e in zip(parameters, result.fitted_parameters, result.errors_percent)
            ],
        }


def batch_fit(
    project: str,
    identifiers: list[str] | None = None,
    dataset_id: str | None = None,
    circuit: str | None = None,
    timeout_seconds: float = 60.0,
    method: str = "least_squares",
    refine_z_threshold: float | None = None,
    refine_max_iterations: int | None = None,
) -> dict:
    """Fit many spectra (selection, one dataset, or the whole project)."""
    with _SESSION.lock:
        opened = _get_project(project)
        targets: list[SpectrumFitTarget] = []
        if identifiers:
            for identifier in identifiers:
                ds_id, loaded, cycle, _s = _resolve(opened, identifier)
                targets.append(
                    SpectrumFitTarget(
                        loaded=loaded,
                        cycle=cycle.cycle,
                        label=f"{loaded.dataset_label}, cycle {cycle.cycle}",
                    )
                )
        else:
            for ds_id, loaded, _state in opened.datasets:
                if dataset_id is not None and ds_id != dataset_id:
                    continue
                for spectrum in loaded.spectra:
                    _ensure_cycle(loaded, spectrum.cycle)
                    targets.append(
                        SpectrumFitTarget(
                            loaded=loaded,
                            cycle=spectrum.cycle,
                            label=f"{loaded.dataset_label}, cycle {spectrum.cycle}",
                        )
                    )
        if not targets:
            raise ValueError("No spectra selected for batch fit")
        first = targets[0]
        initial_parameters = first.loaded.state.parameters_for(first.cycle)
        report = batch_fit_spectra(
            targets,
            initial_parameters,
            use_target_initial_parameters=True,
            initial_circuit=circuit,
            fit_timeout_seconds=float(timeout_seconds),
            fit_options=_fit_options(method),
            refine_z_threshold=refine_z_threshold,
            refine_max_iterations=refine_max_iterations,
        )
        applied = 0
        failed = []
        refined_points = 0
        for result in report.fits:
            if result.fit is None:
                failed.append(result.error or "unknown error")
                continue
            cycle = result.fit.cycle
            cycle.parameters = result.fit.parameters
            cycle.fit_parameters = np.asarray(result.fit.fitted_parameters, dtype=float)
            cycle.fit_frequency_hz = result.fit.fit_frequency_hz
            cycle.fit_impedance = result.fit.fit_impedance
            cycle.fit_at_data_impedance = result.fit.fit_at_data_impedance
            cycle.fit_provenance = dict(result.fit.fit_provenance or {})
            outlier_indices = np.asarray(result.fit.refined_outlier_indices, dtype=int)
            valid = outlier_indices[
                (outlier_indices >= 0) & (outlier_indices < cycle.frequency_hz.size)
            ]
            if valid.size:
                cycle.outliers[valid] = True
                cycle.manually_included[valid] = False
                cycle.invalidate_drt_cache()
                refined_points += int(valid.size)
            result.loaded.state.cycles[cycle.cycle] = cycle
            applied += 1
        return {
            "applied": applied,
            "failed": failed,
            "stopped": bool(report.stopped),
            "skipped": list(report.skipped_labels),
            "refined_outlier_points": refined_points,
        }


def refine_fit(
    project: str,
    identifier: str,
    z_threshold: float = 3.5,
    max_iterations: int = 5,
    timeout_seconds: float | None = None,
) -> dict:
    """Refine one existing fit by iteratively removing high-z residual points."""
    from eis_services import refine_fit_cycle

    with _SESSION.lock:
        opened = _get_project(project)
        dataset_id, loaded, cycle, _spectrum = _resolve(opened, identifier)
        parameters = loaded.state.parameters_for(cycle.cycle)
        fit_result, removed_indices, iterations = refine_fit_cycle(
            cycle,
            cycle.model(loaded.state.circuit),
            parameters,
            float(z_threshold),
            int(max_iterations),
            fit_timeout_seconds=timeout_seconds,
        )
        valid = np.asarray(removed_indices, dtype=int)
        valid = valid[(valid >= 0) & (valid < cycle.frequency_hz.size)]
        cycle.outliers[valid] = True
        cycle.manually_included[valid] = False
        cycle.invalidate_drt_cache()
        _store_fit(cycle, parameters, fit_result)
        loaded.state.cycles[cycle.cycle] = cycle
        return {
            "identity": f"{dataset_id}::cycle-{cycle.cycle}",
            "iterations": int(iterations),
            "removed_points": [int(i) for i in valid],
            "rmse": float(fit_result.rmse),
            "converged": bool(fit_result.converged),
        }


def analyze_outliers(project: str, identifier: str, threshold: float = 1.0) -> dict:
    """Dry-run outlier search for one spectrum; returns proposed point indices."""
    with _SESSION.lock:
        opened = _get_project(project)
        dataset_id, loaded, cycle, _spectrum = _resolve(opened, identifier)
        analysis = analyze_outliers(cycle, float(threshold), loaded.state.parameters_for(cycle.cycle))
        return {
            "identity": f"{dataset_id}::cycle-{cycle.cycle}",
            "threshold": float(threshold),
            "proposed_excluded_indices": [int(i) for i in np.asarray(analysis.outlier_indices, dtype=int)],
            "peak_count": int(analysis.peak_count),
            "ohmic_resistance": float(analysis.ohmic_resistance),
            "inductance": float(analysis.inductance),
            "note": "dry-run only; apply with apply_outliers(confirm=true)",
        }


def hybrid_drt(project: str, identifier: str, max_points: int = 200) -> dict:
    """Compute the hybrid (Bayesian) DRT for one spectrum."""
    with _SESSION.lock:
        opened = _get_project(project)
        dataset_id, loaded, cycle, _spectrum = _resolve(opened, identifier)
        computation = calculate_hybrid_drt(cycle)
        tau = np.asarray(computation.tau_s, dtype=float)
        gamma = np.asarray(computation.gamma_ohm, dtype=float)
        cap = max(0, int(max_points))
        if cap and tau.size > cap:
            step = int(np.ceil(tau.size / cap))
            index = np.arange(0, tau.size, step)
        else:
            index = np.arange(tau.size)
        cycle.saved_hybrid_tau_s = tau
        cycle.saved_hybrid_gamma_ohm = gamma
        cycle.saved_hybrid_ohmic_resistance = computation.ohmic_resistance
        return {
            "identity": f"{dataset_id}::cycle-{cycle.cycle}",
            "ohmic_resistance": computation.ohmic_resistance,
            "points": int(tau.size),
            "tau_s": [float(v) for v in tau[index]],
            "gamma_ohm": [float(v) for v in gamma[index]],
        }


def lin_kk_residuals(project: str, identifier: str, max_points: int = 200) -> dict:
    """Linear Kramers-Kronig residuals for one spectrum."""
    with _SESSION.lock:
        opened = _get_project(project)
        dataset_id, loaded, cycle, _spectrum = _resolve(opened, identifier)
        residuals = calculate_lin_kk_residuals(cycle)
        real = np.asarray(residuals.residual_real, dtype=float)
        imag = np.asarray(residuals.residual_imag, dtype=float)
        mask = np.asarray(cycle.included, dtype=bool)
        cycle.kk_fit_impedance = np.asarray(residuals.fit_impedance, dtype=complex)
        cycle.kk_residual_real = real
        cycle.kk_residual_imag = imag
        cycle.kk_included_mask = mask
        cap = max(0, int(max_points))
        index = np.arange(real.size)
        if cap and real.size > cap:
            step = int(np.ceil(real.size / cap))
            index = np.arange(0, real.size, step)
        return {
            "identity": f"{dataset_id}::cycle-{cycle.cycle}",
            "mean_abs_residual_real": float(np.mean(np.abs(real[mask]))) if mask.any() else None,
            "mean_abs_residual_imag": float(np.mean(np.abs(imag[mask]))) if mask.any() else None,
            "residual_real": [float(v) for v in real[index]],
            "residual_imag": [float(v) for v in imag[index]],
        }


def suggest_eec_parameters(project: str, identifier: str, max_neighbors: int = 5) -> dict:
    """Suggest initial parameters for one spectrum from nearby fitted spectra."""
    from eis_services import suggest_eec_parameters as _suggest

    with _SESSION.lock:
        opened = _get_project(project)
        dataset_id, loaded, cycle, _spectrum = _resolve(opened, identifier)
        records = []
        for ds_id, other, _state in opened.datasets:
            for spectrum in other.spectra:
                other_cycle = _ensure_cycle(other, spectrum.cycle)
                if other_cycle.fit_parameters is None or not other_cycle.parameters:
                    continue
                records.append(
                    FittedEECRecord(
                        identity=f"{ds_id}::cycle-{spectrum.cycle}",
                        cycle=int(spectrum.cycle),
                        potential_v=float(spectrum.potential_v),
                        current_ma=float(spectrum.current_ma),
                        circuit=other_cycle.model(other.state.circuit),
                        parameter_names=tuple(p.name for p in other_cycle.parameters),
                        fitted_parameters=tuple(
                            float(v) for v in np.asarray(other_cycle.fit_parameters, dtype=float)
                        ),
                        custom_metadata=dict(spectrum.custom_metadata),
                        project_id=opened.path.name,
                    )
                )
        target = next(
            (
                r
                for r in records
                if r.identity == f"{dataset_id}::cycle-{cycle.cycle}"
            ),
            None,
        )
        if target is None:
            target = FittedEECRecord(
                identity=f"{dataset_id}::cycle-{cycle.cycle}",
                cycle=cycle.cycle,
                potential_v=float(cycle.potential_v),
                current_ma=float(cycle.current_ma),
                circuit=cycle.model(loaded.state.circuit),
                parameter_names=tuple(p.name for p in cycle.parameters),
                fitted_parameters=tuple(p.initial for p in cycle.parameters),
                custom_metadata={},
                project_id=opened.path.name,
            )
        suggestion = _suggest(target, records, max_neighbors=int(max_neighbors))
        return {
            "identity": target.identity,
            "values": {k: float(v) for k, v in suggestion.values.items()},
            "contributors": [
                {"identity": c.identity, "cycle": c.cycle, "distance": c.distance}
                for c in suggestion.contributors
            ],
            "candidate_count": suggestion.candidate_count,
            "usable_candidate_count": suggestion.usable_candidate_count,
            "warnings": list(suggestion.warnings),
        }


# ---------------------------------------------------------------------------
# Tools: gated mutations (dry-run unless confirm=true)
# ---------------------------------------------------------------------------


def _parameter_changes(cycle: CycleState, changes: list[dict]) -> tuple[list[ParameterChange], list[str]]:
    by_name = {p.name: p for p in cycle.parameters}
    edits: list[ParameterChange] = []
    errors: list[str] = []
    for change in changes:
        if not isinstance(change, dict):
            errors.append(f"change entry must be an object: {change!r}")
            continue
        name = str(change.get("name") or "")
        parameter = by_name.get(name)
        if parameter is None:
            errors.append(f"unknown parameter {name!r}")
            continue
        for field_name in ("initial", "lower", "upper", "fixed"):
            if field_name not in change or change[field_name] is None:
                continue
            new_value = change[field_name]
            if field_name == "fixed":
                new_value = bool(new_value)
            else:
                try:
                    new_value = float(new_value)
                except (TypeError, ValueError):
                    errors.append(f"{name}.{field_name}: not a number: {new_value!r}")
                    continue
                if not np.isfinite(new_value):
                    errors.append(f"{name}.{field_name}: must be finite")
                    continue
            edits.append(ParameterChange(name=name, field=field_name, old=getattr(parameter, field_name), new=new_value))
    return edits, errors


def _validate_bounds(parameters: list[ParameterValue]) -> str | None:
    for parameter in parameters:
        if parameter.lower > parameter.upper:
            return f"{parameter.name}: lower > upper"
        if not parameter.lower <= parameter.initial <= parameter.upper:
            return f"{parameter.name}: initial outside [lower, upper]"
    return None


def set_parameters(
    project: str,
    identifier: str,
    changes: list[dict],
    confirm: bool = False,
) -> dict:
    """Change parameter initial/lower/upper/fixed of one spectrum.

    ``changes`` is a list of ``{"name": ..., "initial"/"lower"/"upper"/"fixed": ...}``.
    Dry-run by default: returns the proposed diff.  With ``confirm=true`` the
    change is applied (fit cleared) and logged to the user-actions sidecar.
    """
    with _SESSION.lock:
        opened = _get_project(project)
        dataset_id, loaded, cycle, _spectrum = _resolve(opened, identifier)
        parameters = loaded.state.parameters_for(cycle.cycle)
        by_name = {p.name: p for p in parameters}
        edits, errors = _parameter_changes(cycle, changes)
        if errors:
            return {"ok": False, "errors": errors}
        proposal = []
        working = {
            p.name: ParameterValue(p.name, p.unit, p.initial, p.lower, p.upper, p.error_percent, p.fixed)
            for p in parameters
        }
        for edit in edits:
            setattr(working[edit.name], edit.field, edit.new)
            proposal.append(edit.to_dict())
        bound_error = _validate_bounds(list(working.values()))
        if bound_error is not None:
            return {"ok": False, "errors": [f"invalid bounds: {bound_error}"], "changes": proposal}
        if not confirm:
            return {
                "ok": True,
                "dry_run": True,
                "confirm_required": True,
                "identity": f"{dataset_id}::cycle-{cycle.cycle}",
                "changes": proposal,
            }
        for edit in edits:
            setattr(by_name[edit.name], edit.field, edit.new)
        cycle.parameters = parameters
        cycle.clear_fit()
        loaded.state.cycles[cycle.cycle] = cycle
        change = CycleChange(cycle=str(cycle.cycle), parameters=edits)
        _log_action(opened, dataset_id, loaded, cycle, "set parameters", change)
        return {
            "ok": True,
            "applied": True,
            "identity": f"{dataset_id}::cycle-{cycle.cycle}",
            "changes": proposal,
            "fit_cleared": True,
        }


def set_frequency_window(
    project: str,
    identifier: str,
    fmin: float,
    fmax: float,
    confirm: bool = False,
) -> dict:
    """Set the frequency window of one spectrum.  Dry-run unless confirm=true."""
    with _SESSION.lock:
        opened = _get_project(project)
        dataset_id, loaded, cycle, _spectrum = _resolve(opened, identifier)
        fmin, fmax = float(fmin), float(fmax)
        if not np.isfinite(fmin) or not np.isfinite(fmax) or fmin <= 0 or fmax <= 0:
            return {"ok": False, "errors": ["fmin/fmax must be positive finite numbers"]}
        window = tuple(sorted((fmin, fmax)))
        old = list(cycle.frequency_window) if cycle.frequency_window is not None else None
        diff = {"old": old, "new": list(window)}
        if not confirm:
            return {
                "ok": True,
                "dry_run": True,
                "confirm_required": True,
                "identity": f"{dataset_id}::cycle-{cycle.cycle}",
                "frequency_window": diff,
            }
        cycle.frequency_window = window
        cycle.invalidate_drt_cache()
        cycle.clear_fit()
        loaded.state.cycles[cycle.cycle] = cycle
        change = CycleChange(cycle=str(cycle.cycle), frequency_window=(old, list(window)))
        _log_action(opened, dataset_id, loaded, cycle, "set frequency window", change)
        return {
            "ok": True,
            "applied": True,
            "identity": f"{dataset_id}::cycle-{cycle.cycle}",
            "frequency_window": diff,
            "fit_cleared": True,
        }


def set_point_mask(
    project: str,
    identifier: str,
    exclude: list[int] | None = None,
    include: list[int] | None = None,
    confirm: bool = False,
) -> dict:
    """Exclude/include points of one spectrum by index.  Dry-run unless confirm=true."""
    with _SESSION.lock:
        opened = _get_project(project)
        dataset_id, loaded, cycle, _spectrum = _resolve(opened, identifier)
        size = cycle.frequency_hz.size
        exclude = [int(i) for i in (exclude or [])]
        include = [int(i) for i in (include or [])]
        invalid = [i for i in exclude + include if not 0 <= i < size]
        if invalid:
            return {"ok": False, "errors": [f"point index out of range (0..{size - 1}): {invalid}"]}
        old_mask = np.asarray(cycle.manually_included, dtype=bool).copy()
        new_mask = old_mask.copy()
        new_mask[exclude] = False
        new_mask[include] = True
        diff = {
            "points_excluded": [int(i) for i in np.nonzero(old_mask & ~new_mask)[0]],
            "points_included": [int(i) for i in np.nonzero(~old_mask & new_mask)[0]],
        }
        if np.array_equal(old_mask, new_mask):
            return {"ok": True, "dry_run": not confirm, "identity": f"{dataset_id}::cycle-{cycle.cycle}", "changes": []}
        if not confirm:
            return {
                "ok": True,
                "dry_run": True,
                "confirm_required": True,
                "identity": f"{dataset_id}::cycle-{cycle.cycle}",
                "changes": diff,
            }
        cycle.manually_included = new_mask
        cycle.outliers[exclude] = True
        cycle.outliers[include] = False
        cycle.invalidate_drt_cache()
        cycle.clear_fit()
        loaded.state.cycles[cycle.cycle] = cycle
        change = CycleChange(
            cycle=str(cycle.cycle),
            points_excluded=diff["points_excluded"],
            points_included=diff["points_included"],
        )
        _log_action(opened, dataset_id, loaded, cycle, "set point mask", change)
        return {
            "ok": True,
            "applied": True,
            "identity": f"{dataset_id}::cycle-{cycle.cycle}",
            "changes": diff,
            "fit_cleared": True,
        }


def apply_outliers(
    project: str,
    identifier: str,
    indices: list[int],
    confirm: bool = False,
) -> dict:
    """Mark points as outliers (excluded from fits).  Dry-run unless confirm=true."""
    with _SESSION.lock:
        opened = _get_project(project)
        dataset_id, loaded, cycle, _spectrum = _resolve(opened, identifier)
        size = cycle.frequency_hz.size
        valid = [int(i) for i in indices if 0 <= int(i) < size]
        invalid = [int(i) for i in indices if not 0 <= int(i) < size]
        if not confirm:
            return {
                "ok": True,
                "dry_run": True,
                "confirm_required": True,
                "identity": f"{dataset_id}::cycle-{cycle.cycle}",
                "proposed_excluded_indices": valid,
                "ignored_invalid_indices": invalid,
            }
        cycle.apply_outliers(np.asarray(valid, dtype=int))
        loaded.state.cycles[cycle.cycle] = cycle
        change = CycleChange(cycle=str(cycle.cycle), points_excluded=valid)
        _log_action(opened, dataset_id, loaded, cycle, "apply outliers", change)
        return {
            "ok": True,
            "applied": True,
            "identity": f"{dataset_id}::cycle-{cycle.cycle}",
            "excluded_indices": valid,
            "ignored_invalid_indices": invalid,
            "fit_cleared": True,
        }


# ---------------------------------------------------------------------------
# Tools: correction memory (flag bad fits, propose and apply replayed fixes)
# ---------------------------------------------------------------------------


def _context_snapshots(opened: _OpenProject, dataset_ids: set[str] | None):
    rows = [
        (dataset_id, loaded, spectrum)
        for dataset_id, loaded, _state in opened.datasets
        if dataset_ids is None or dataset_id in dataset_ids
        for spectrum in loaded.spectra
    ]
    return rows, [_snapshot(dataset_id, loaded, spectrum) for dataset_id, loaded, spectrum in rows]


def flag_bad_fits(
    project: str,
    z_threshold: float = 4.0,
    residual_threshold: float | None = None,
    dataset_id: str | None = None,
) -> dict:
    """Flag badly fitted spectra via robust local z-scores over neighbor fits."""
    with _SESSION.lock:
        opened = _get_project(project)
        _rows, snapshots = _context_snapshots(opened, {dataset_id} if dataset_id else None)
        features = effective_features(snapshots, DEFAULT_SIMILARITY_FEATURES)
        flags = _flag_bad_fits_impl(
            snapshots,
            z_threshold=float(z_threshold),
            feature_names=features,
            residual_threshold=residual_threshold,
        )
        return {
            "project": opened.key,
            "z_threshold": float(z_threshold),
            "features": list(features),
            "flagged": [
                {
                    "identity": flag.snapshot.identity,
                    "spectrum_id": flag.snapshot.spectrum_id,
                    "dataset_id": flag.snapshot.dataset_id,
                    "cycle": flag.snapshot.cycle,
                    "reasons": list(flag.reasons),
                    "max_abs_z": flag.max_abs_z,
                    "residual": flag.residual,
                    "worst_parameters": [
                        {"name": d.name, "actual": d.actual, "expected": d.expected, "robust_z": d.robust_z}
                        for d in flag.parameters[:5]
                    ],
                }
                for flag in flags
            ],
        }


def propose_corrections(
    project: str,
    z_threshold: float = 4.0,
    max_sources: int = 1,
    identifiers: list[str] | None = None,
) -> dict:
    """Propose replayed corrections for badly fitted spectra.

    Reads the project's ``_user_actions.json`` sidecar (GUI-recorded or
    agent-applied fixes) and proposes the nearest structurally compatible
    logged parameter/circuit changes.  Never proposes point masks or frequency
    windows.  Apply with ``apply_corrections(confirm=true)``.
    """
    with _SESSION.lock:
        opened = _get_project(project)
        dataset_ids = None
        selected = None
        if identifiers:
            resolved = [_resolve(opened, identifier) for identifier in identifiers]
            dataset_ids = {dataset_id for dataset_id, _l, _c, _s in resolved}
            selected = {(dataset_id, cycle.cycle) for dataset_id, _l, cycle, _s in resolved}
        _rows, snapshots = _context_snapshots(opened, dataset_ids)
        features = effective_features(snapshots, DEFAULT_SIMILARITY_FEATURES)
        flags = _flag_bad_fits_impl(snapshots, z_threshold=float(z_threshold), feature_names=features)
        if selected is not None:
            flags = [f for f in flags if (f.snapshot.dataset_id, f.snapshot.cycle) in selected]
        entries = load_user_actions(user_actions_path_for(opened.path))
        proposals = _propose_corrections_impl(flags, entries, feature_names=features, max_sources=int(max_sources))
        return {
            "project": opened.key,
            "z_threshold": float(z_threshold),
            "logged_corrections": len(correction_entries(entries)),
            "proposals": [_proposal_dict(p) for p in proposals],
        }


def apply_corrections(
    project: str,
    proposal_identities: list[str] | None = None,
    z_threshold: float = 4.0,
    max_sources: int = 1,
    confirm: bool = False,
    refit: bool = False,
    timeout_seconds: float = 60.0,
) -> dict:
    """Apply replayed corrections proposed by ``propose_corrections``.

    Dry-run by default: returns the proposed changes per spectrum.  With
    ``confirm=true`` the parameter/circuit changes are applied (bounds
    validated, fit cleared), logged to the sidecar as ``applied_by: "mcp"``
    with their replay sources, and optionally refitted when ``refit=true``.
    Pass ``proposal_identities`` (from ``propose_corrections``) to apply a
    subset; omit to apply all proposals.
    """
    with _SESSION.lock:
        opened = _get_project(project)
        _rows, snapshots = _context_snapshots(opened, None)
        features = effective_features(snapshots, DEFAULT_SIMILARITY_FEATURES)
        flags = _flag_bad_fits_impl(snapshots, z_threshold=float(z_threshold), feature_names=features)
        entries = load_user_actions(user_actions_path_for(opened.path))
        proposals = _propose_corrections_impl(flags, entries, feature_names=features, max_sources=int(max_sources))
        if proposal_identities:
            wanted = {str(i) for i in proposal_identities}
            proposals = [p for p in proposals if p.identity in wanted]
        dry_run = [
            {
                "identity": p.identity,
                "dataset_id": p.snapshot.dataset_id,
                "cycle": p.snapshot.cycle,
                "reasons": list(p.flag.reasons),
                "changes": [dict(c) for c in p.changes],
                "sources": [dict(s) for s in p.sources],
            }
            for p in proposals
        ]
        if not confirm:
            return {
                "ok": True,
                "dry_run": True,
                "confirm_required": True,
                "project": opened.key,
                "proposals": dry_run,
            }
        applied = []
        skipped = []
        targets: list[SpectrumFitTarget] = []
        for proposal in proposals:
            snapshot = proposal.snapshot
            loaded = next((l for d, l, _s in opened.datasets if d == snapshot.dataset_id), None)
            if loaded is None:
                skipped.append(snapshot.identity)
                continue
            cycle = _ensure_cycle(loaded, snapshot.cycle)
            parameters = loaded.state.parameters_for(snapshot.cycle)
            by_name = {p.name: p for p in parameters}
            changed = False
            circuit_change = None
            for change in proposal.changes:
                field = str(change.get("field"))
                if field == "circuit":
                    new_circuit = str(change.get("new") or "")
                    if new_circuit and circuits_equivalent(cycle.model(loaded.state.circuit), new_circuit):
                        cycle.circuit = new_circuit
                        circuit_change = (snapshot.circuit, new_circuit)
                        changed = True
                    continue
                parameter = by_name.get(str(change.get("name")))
                if parameter is None or change.get("new") is None:
                    continue
                if field == "initial":
                    parameter.initial = float(change["new"])
                elif field == "lower":
                    parameter.lower = float(change["new"])
                elif field == "upper":
                    parameter.upper = float(change["new"])
                elif field == "fixed":
                    parameter.fixed = bool(change["new"])
                else:
                    continue
                changed = True
            if not changed or _validate_bounds(parameters) is not None:
                skipped.append(snapshot.identity)
                continue
            cycle.parameters = parameters
            cycle.clear_fit()
            loaded.state.cycles[cycle.cycle] = cycle
            change = CycleChange(
                cycle=str(cycle.cycle),
                parameters=[
                    ParameterChange(
                        name=str(c.get("name")),
                        field=str(c.get("field", "initial")),
                        old=c.get("old"),
                        new=c.get("new"),
                    )
                    for c in proposal.changes
                    if c.get("field") != "circuit"
                ],
                circuit=circuit_change,
            )
            _log_action(
                opened,
                snapshot.dataset_id,
                loaded,
                cycle,
                "apply replayed correction",
                change,
                replay_sources=proposal.sources,
            )
            applied.append(
                {
                    "identity": snapshot.identity,
                    "changes": [dict(c) for c in proposal.changes],
                    "sources": [dict(s) for s in proposal.sources],
                }
            )
            if refit:
                targets.append(
                    SpectrumFitTarget(
                        loaded=loaded,
                        cycle=cycle.cycle,
                        label=f"{loaded.dataset_label}, cycle {cycle.cycle}",
                    )
                )
        refit_report = None
        if refit and targets:
            report = batch_fit_spectra(
                targets,
                targets[0].loaded.state.parameters_for(targets[0].cycle),
                use_target_initial_parameters=True,
                fit_timeout_seconds=float(timeout_seconds),
                fit_options=_fit_options("least_squares"),
            )
            fitted = 0
            for result in report.fits:
                if result.fit is None:
                    continue
                cycle = result.fit.cycle
                cycle.parameters = result.fit.parameters
                cycle.fit_parameters = np.asarray(result.fit.fitted_parameters, dtype=float)
                cycle.fit_frequency_hz = result.fit.fit_frequency_hz
                cycle.fit_impedance = result.fit.fit_impedance
                cycle.fit_at_data_impedance = result.fit.fit_at_data_impedance
                cycle.fit_provenance = dict(result.fit.fit_provenance or {})
                result.loaded.state.cycles[cycle.cycle] = cycle
                fitted += 1
            refit_report = {"refitted": fitted, "stopped": bool(report.stopped)}
        return {
            "ok": True,
            "applied": applied,
            "skipped": skipped,
            "refit": refit_report,
        }


# ---------------------------------------------------------------------------
# Tools: persistence
# ---------------------------------------------------------------------------


def save_project(project: str, path: str | None = None, backup: bool = True) -> dict:
    """Save the in-memory project atomically (optionally to a new path).

    When the target file already exists, the previous version is copied to
    ``<file>.bak`` first (unless ``backup=false``).
    """
    import shutil

    with _SESSION.lock:
        opened = _get_project(project)
        target = Path(str(path)).expanduser().resolve() if path else opened.path
        had_previous = target.is_file()
        if backup and had_previous:
            shutil.copy2(target, target.with_name(target.name + ".bak"))
        datasets = [
            (dataset_id, loaded.state, loaded.dataframe)
            for dataset_id, loaded, _state in opened.datasets
        ]
        first = opened.datasets[0]
        save_project_file(first[2], target, datasets=datasets)
        opened.path = target
        opened.stamp = _stamp(target)
        _SESSION.projects.pop(str(Path(str(project)).expanduser().resolve()), None)
        _SESSION.projects[str(target)] = opened
        return {
            "saved": str(target),
            "backup": str(target) + ".bak" if backup and had_previous else None,
            "datasets": [d for d, _l, _s in datasets],
        }


def export_fit_parameters(
    project: str,
    path: str | None = None,
    dataset_id: str | None = None,
) -> dict:
    """Export fitted EEC parameters of a project to a CSV file.

    Writes one row per fitted spectrum with source metadata, all fitted
    parameter values and their error percentages, and derived per-block
    tau/C columns — the same format the GUI's Export menu produces.
    When ``path`` is omitted, defaults to ``<project>_fit_export.csv``
    next to the project file.  Pass ``dataset_id`` to export only one
    dataset.  Returns the output path and the number of rows written.
    """
    with _SESSION.lock:
        opened = _get_project(project)
        states = [loaded.state for _ds, loaded, _st in opened.datasets
                  if dataset_id is None or _ds == dataset_id]
        if not states:
            raise ValueError(f"No datasets matched in project {project!r}")
        target = Path(str(path)).expanduser().resolve() if path else \
            opened.path.with_name(opened.path.stem.replace(".eisfit.json", "") + "_fit_export.csv")
        if target.suffix != ".csv":
            target = target.with_suffix(".csv")
        count = export_fit_parameters_for_states(states, target)
        return {"exported": str(target), "rows": count, "project": str(opened.path)}


# ---------------------------------------------------------------------------
# MCP wiring (lazy import so the tool functions stay testable without the SDK)
# ---------------------------------------------------------------------------

_TOOLS = [
    open_project,
    list_projects,
    close_project,
    reload_project,
    list_spectra,
    get_spectrum,
    get_parameters,
    get_fit,
    list_user_actions,
    fit_spectrum,
    batch_fit,
    refine_fit,
    analyze_outliers,
    hybrid_drt,
    lin_kk_residuals,
    suggest_eec_parameters,
    set_parameters,
    set_frequency_window,
    set_point_mask,
    apply_outliers,
    flag_bad_fits,
    propose_corrections,
    apply_corrections,
    save_project,
    export_fit_parameters,
]


def _list_projects_resource() -> str:
    return json.dumps(list_projects(), indent=2)


def _read_user_actions_resource(name: str) -> str:
    with _SESSION.lock:
        for opened in _SESSION.projects.values():
            if opened.path.name == name:
                sidecar = user_actions_path_for(opened.path)
                if sidecar.is_file():
                    return sidecar.read_text(encoding="utf-8")
                return json.dumps(
                    {"format": "eis-fitting-user-actions", "version": 1, "entries": []}
                )
        raise ValueError(f"No open project named {name!r}")


def _correct_bad_fits_prompt(project: str, z_threshold: float = 4.0) -> str:
    return (
        f"Correct badly fitted spectra in the eisyFIT project '{project}' (z_threshold={z_threshold}):\n"
        "1. Call list_spectra(project) to get spectrum identities.\n"
        "2. Call flag_bad_fits(project, z_threshold=z_threshold) and review the flagged spectra.\n"
        "3. Call propose_corrections(project, z_threshold=z_threshold) and review each proposal's\n"
        "   changes and replay sources; never apply a proposal blindly.\n"
        "4. Call apply_corrections(project, proposal_identities=[...], confirm=true, refit=true)\n"
        "   to apply selected corrections and refit. Omit proposal_identities to apply all proposals.\n"
        "5. Call save_project(project) to persist. Applied corrections are logged to the\n"
        "   _user_actions.json sidecar with applied_by='mcp'."
    )


def build_server():
    """Build the MCP server object (``mcp`` SDK v2 ``MCPServer`` or v1 ``FastMCP``)."""
    try:
        from mcp.server.mcpserver import MCPServer as Server
    except ImportError:  # mcp SDK 1.x
        from mcp.server.fastmcp import FastMCP as Server

    server = Server(
        "eisyfit",
        instructions=(
            "eisyFIT EIS project server. Open a project with open_project(path), inspect with "
            "list_spectra/get_spectrum/get_fit, compute with fit_spectrum/batch_fit/hybrid_drt/"
            "analyze_outliers, and correct bad fits with flag_bad_fits/propose_corrections/"
            "apply_corrections. Mutating tools are dry-run unless confirm=true."
        ),
    )
    for function in _TOOLS:
        server.tool(name=function.__name__, description=function.__doc__)(function)
    server.resource(
        "eisyfit://projects",
        name="open_projects",
        description="JSON list of currently open eisyFIT projects",
        mime_type="application/json",
    )(_list_projects_resource)
    server.resource(
        "eisyfit://projects/{name}/user-actions",
        name="user_actions",
        description="Recorded user/agent corrections sidecar of an open project (file name as {name})",
        mime_type="application/json",
    )(_read_user_actions_resource)
    server.prompt(
        name="correct_bad_fits",
        description="Workflow: flag bad fits, review replayed corrections, apply with confirmation, refit, save",
    )(_correct_bad_fits_prompt)
    return server


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="eisyfit-mcp",
        description="MCP server for eisyFIT EIS projects (stdio transport).",
    )
    parser.add_argument(
        "--transport",
        choices=["stdio"],
        default="stdio",
        help="MCP transport; only stdio is supported (default).",
    )
    parser.add_argument(
        "--project",
        type=str,
        default=None,
        help="Optional project or data file to open at startup.",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING)
    if args.project:
        open_project(args.project)
    server = build_server()
    server.run(transport=args.transport)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Sidecar log of the user's manual correction steps.

The log is stored next to the project as ``<project>_user_actions.json`` and
uses the versioned format ``eis-fitting-user-actions`` / ``1``.  Entries are
keyed by the path-independent :func:`ml.results_schema.spectrum_identifier`
so the log stays valid when project files are moved or renamed.  Raw measured
arrays are never duplicated here (only the hash key and metadata snapshots),
following the same sidecar rules as the ML results files.

The GUI appends one entry per detected state change while recording is
enabled in Preferences; entries written by the correction-replay feature are
tagged with ``applied_by: "replay"`` so user-taught and replayed steps can be
distinguished later.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

USER_ACTIONS_FORMAT = "eis-fitting-user-actions"
USER_ACTIONS_VERSION = 1
MAX_USER_ACTIONS = 5000

# Parameter fields a user can edit in the parameter table.
EDITABLE_PARAMETER_FIELDS = ("initial", "lower", "upper", "fixed")


@dataclass
class ParameterChange:
    """One edited parameter field, e.g. initial value or a bound."""

    name: str
    field: str
    old: object
    new: object

    def to_dict(self) -> dict[str, object]:
        return {"name": self.name, "field": self.field, "old": self.old, "new": self.new}


@dataclass
class CycleChange:
    """Structured diff of one cycle's serialized state."""

    cycle: str
    parameters: list[ParameterChange] = field(default_factory=list)
    circuit: tuple[str | None, str | None] | None = None
    frequency_window: tuple[object, object] | None = None
    points_excluded: list[int] = field(default_factory=list)
    points_included: list[int] = field(default_factory=list)
    fit_changed: bool = False

    @property
    def is_empty(self) -> bool:
        return not (
            self.parameters
            or self.circuit is not None
            or self.frequency_window is not None
            or self.points_excluded
            or self.points_included
            or self.fit_changed
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "cycle": self.cycle,
            "parameters": [change.to_dict() for change in self.parameters],
            "circuit": (
                {"old": self.circuit[0], "new": self.circuit[1]}
                if self.circuit is not None
                else None
            ),
            "frequency_window": (
                {"old": self.frequency_window[0], "new": self.frequency_window[1]}
                if self.frequency_window is not None
                else None
            ),
            "points_excluded": list(self.points_excluded),
            "points_included": list(self.points_included),
            "fit_changed": self.fit_changed,
        }


def user_actions_path_for(project_path: Path) -> Path:
    """Return the sidecar path belonging to a ``.eisfit.json[.gz]`` project."""
    path = Path(project_path)
    name = path.name
    for suffix in (".gz", ".json", ".eisfit"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    return path.with_name(f"{name}_user_actions.json")


def diff_parameters(
    old_parameters: object, new_parameters: object
) -> list[ParameterChange]:
    """Compare two serialized parameter lists (``_parameter_to_dict`` output)."""
    old_list = old_parameters if isinstance(old_parameters, list) else []
    new_list = new_parameters if isinstance(new_parameters, list) else []
    old_by_name = {
        str(item.get("name")): item for item in old_list if isinstance(item, dict)
    }
    new_by_name = {
        str(item.get("name")): item for item in new_list if isinstance(item, dict)
    }
    changes: list[ParameterChange] = []
    for name in sorted(set(old_by_name) | set(new_by_name)):
        old_item = old_by_name.get(name)
        new_item = new_by_name.get(name)
        if old_item is None or new_item is None:
            continue
        for field_name in EDITABLE_PARAMETER_FIELDS:
            old_value = old_item.get(field_name)
            new_value = new_item.get(field_name)
            if old_value != new_value:
                changes.append(
                    ParameterChange(
                        name=name,
                        field=field_name,
                        old=old_value,
                        new=new_value,
                    )
                )
    return changes


def _mask_indices(old_mask: object, new_mask: object) -> tuple[list[int], list[int]]:
    old_array = np.asarray(old_mask, dtype=bool) if old_mask is not None else None
    new_array = np.asarray(new_mask, dtype=bool) if new_mask is not None else None
    if old_array is None or new_array is None or old_array.shape != new_array.shape:
        return [], []
    excluded = [int(i) for i in np.nonzero(old_array & ~new_array)[0]]
    included = [int(i) for i in np.nonzero(~old_array & new_array)[0]]
    return excluded, included


def diff_cycle_dict(old_cycle: dict, new_cycle: dict) -> CycleChange | None:
    """Build the structured change between two serialized cycle dicts.

    Returns ``None`` when nothing user-relevant changed.  Only keys that a
    user action can touch are considered (parameters, circuit, frequency
    window, point inclusion, fit presence); DRT/KK caches and metadata are
    ignored on purpose.
    """
    if not isinstance(old_cycle, dict) or not isinstance(new_cycle, dict):
        return None
    change = CycleChange(cycle="")
    change.parameters = diff_parameters(
        old_cycle.get("parameters"), new_cycle.get("parameters")
    )
    old_circuit = old_cycle.get("circuit")
    new_circuit = new_cycle.get("circuit")
    if old_circuit != new_circuit:
        change.circuit = (old_circuit, new_circuit)
    old_window = old_cycle.get("frequency_window")
    new_window = new_cycle.get("frequency_window")
    if old_window != new_window:
        change.frequency_window = (old_window, new_window)
    excluded, included = _mask_indices(
        old_cycle.get("manually_included"), new_cycle.get("manually_included")
    )
    change.points_excluded = excluded
    change.points_included = included
    old_outliers = np.asarray(
        old_cycle.get("outliers") or [], dtype=bool
    )
    new_outliers = np.asarray(
        new_cycle.get("outliers") or [], dtype=bool
    )
    if old_outliers.shape == new_outliers.shape and old_outliers.size:
        change.points_excluded = sorted(
            set(change.points_excluded)
            | {int(i) for i in np.nonzero(new_outliers & ~old_outliers)[0]}
        )
    change.fit_changed = (old_cycle.get("fit_parameters") is None) != (
        new_cycle.get("fit_parameters") is None
    ) or old_cycle.get("fit_provenance") != new_cycle.get("fit_provenance")
    return None if change.is_empty else change


def diff_state_payloads(
    old_payload: object, new_payload: object
) -> list[dict[str, object]]:
    """Structured per-dataset, per-cycle diff of two light project payloads.

    Mirrors :func:`eis_project.describe_history_change` but returns data
    instead of a label string.  Each returned item is
    ``{"dataset_id": ..., "cycles": [CycleChange.to_dict(), ...]}``; entries
    without changes are omitted.
    """
    if not isinstance(old_payload, list) or not isinstance(new_payload, list):
        return []
    old_by_id = {
        str(dataset.get("dataset_id")): dataset
        for dataset in old_payload
        if isinstance(dataset, dict)
    }
    results: list[dict[str, object]] = []
    for new_dataset in new_payload:
        if not isinstance(new_dataset, dict):
            continue
        dataset_id = str(new_dataset.get("dataset_id"))
        old_dataset = old_by_id.get(dataset_id)
        old_state = old_dataset.get("state", {}) if isinstance(old_dataset, dict) else {}
        new_state = new_dataset.get("state", {})
        if not isinstance(old_state, dict) or not isinstance(new_state, dict):
            continue
        old_cycles = old_state.get("cycles", {})
        new_cycles = new_state.get("cycles", {})
        if not isinstance(old_cycles, dict) or not isinstance(new_cycles, dict):
            continue
        cycle_changes: list[dict[str, object]] = []
        for cycle_text in sorted(set(old_cycles) | set(new_cycles), key=str):
            old_cycle = old_cycles.get(cycle_text)
            new_cycle = new_cycles.get(cycle_text)
            if old_cycle == new_cycle:
                continue
            if not isinstance(old_cycle, dict) or not isinstance(new_cycle, dict):
                continue
            change = diff_cycle_dict(old_cycle, new_cycle)
            if change is None:
                continue
            change.cycle = str(cycle_text)
            cycle_changes.append(change.to_dict())
        if cycle_changes:
            results.append({"dataset_id": dataset_id, "cycles": cycle_changes})
    return results


def make_action_entry(
    *,
    applied_by: str,
    action: str,
    dataset_id: str,
    dataset_label: str,
    control: str,
    spectrum_id: str | None,
    cycle: int | str | None,
    circuit: str | None,
    metadata: dict[str, object] | None,
    change: CycleChange | dict[str, object] | None,
    residual_before: float | None = None,
    residual_after: float | None = None,
    timestamp: str | None = None,
) -> dict[str, object]:
    """Assemble one sidecar entry with a stable schema."""
    if timestamp is None:
        timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    change_dict = change.to_dict() if isinstance(change, CycleChange) else (change or {})
    return {
        "timestamp": timestamp,
        "applied_by": applied_by if applied_by in {"user", "replay"} else "user",
        "action": str(action),
        "dataset_id": str(dataset_id),
        "dataset_label": str(dataset_label),
        "control": str(control),
        "spectrum_id": spectrum_id,
        "cycle": cycle,
        "circuit": circuit,
        "metadata": dict(metadata or {}),
        "changes": change_dict,
        "residual_before": residual_before,
        "residual_after": residual_after,
    }


def write_user_actions(
    path: Path,
    entries: list[dict],
    *,
    source_project: str | None = None,
    similarity_features: list[str] | None = None,
) -> None:
    """Write the sidecar atomically (temporary file + replace)."""
    path = Path(path)
    payload: dict[str, object] = {
        "format": USER_ACTIONS_FORMAT,
        "version": USER_ACTIONS_VERSION,
        "source_project": source_project,
        "similarity_features": list(similarity_features or []),
        "entries": list(entries),
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def load_user_actions_payload(payload: dict) -> list[dict]:
    """Validate a sidecar payload and return its entries.

    Raises ``ValueError`` for a foreign format or an unsupported newer
    version; malformed individual entries are dropped, never fatal.
    """
    if not isinstance(payload, dict):
        raise ValueError("user actions payload must be a JSON object")
    if payload.get("format") != USER_ACTIONS_FORMAT:
        raise ValueError(
            f"not a {USER_ACTIONS_FORMAT} file: {payload.get('format')!r}"
        )
    version = int(payload.get("version", 0))
    if version < 1 or version > USER_ACTIONS_VERSION:
        raise ValueError(f"unsupported user actions version: {version}")
    entries = payload.get("entries", [])
    if not isinstance(entries, list):
        return []
    return [entry for entry in entries if isinstance(entry, dict)]


def load_user_actions(path: Path) -> list[dict]:
    """Load entries from a sidecar; a missing file simply means no log yet."""
    path = Path(path)
    if not path.is_file():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    return load_user_actions_payload(payload)


def append_user_actions(
    path: Path,
    new_entries: list[dict],
    *,
    limit: int = MAX_USER_ACTIONS,
    source_project: str | None = None,
    similarity_features: list[str] | None = None,
) -> None:
    """Append entries to the sidecar, trimming the oldest beyond ``limit``."""
    existing = load_user_actions(path)
    entries = existing + [entry for entry in new_entries if isinstance(entry, dict)]
    if len(entries) > limit:
        entries = entries[-limit:]
    write_user_actions(
        path,
        entries,
        source_project=source_project,
        similarity_features=similarity_features,
    )


def correction_entries(entries: list[dict]) -> list[dict]:
    """Return only entries that changed something replayable.

    A replayable entry carries parameter, circuit, frequency-window, or
    point-selection changes on a known spectrum.
    """
    selected = []
    for entry in entries:
        changes = entry.get("changes")
        if not isinstance(changes, dict):
            continue
        has_change = bool(
            changes.get("parameters")
            or changes.get("circuit") is not None
            or changes.get("frequency_window") is not None
            or changes.get("points_excluded")
            or changes.get("points_included")
        )
        if has_change and entry.get("spectrum_id"):
            selected.append(entry)
    return selected


def relative_fit_residual(
    frequency_hz: np.ndarray,
    impedance: np.ndarray,
    fit_at_data: np.ndarray | None,
    included: np.ndarray | None,
) -> float | None:
    """Mean relative |Z| residual of a fit at the data points.

    Returns ``None`` when there is no fit at the data points or no included
    points.  Used both for the action log and for bad-fit flagging.
    """
    if fit_at_data is None:
        return None
    z = np.asarray(impedance, dtype=complex).reshape(-1)
    fit = np.asarray(fit_at_data, dtype=complex).reshape(-1)
    if z.size == 0 or fit.size != z.size:
        return None
    mask = (
        np.ones(z.size, dtype=bool)
        if included is None
        else np.asarray(included, dtype=bool).reshape(-1)
    )
    if mask.size != z.size or not mask.any():
        return None
    scale = np.maximum(np.abs(z[mask]), np.finfo(float).eps)
    return float(np.mean(np.abs(fit[mask] - z[mask]) / scale))

"""Replay logged manual corrections onto badly fitted spectra.

This module is the "learn from the user's behaviour" half of the correction
memory: no model is trained.  Instead, each badly fitted spectrum is compared
with the spectra the user previously corrected, using a configurable set of
similarity features (by default time, voltage, current, cycle number, and
cycle mod), and the parameter changes from the nearest logged corrections are
proposed, mapped structurally when the EEC topology differs.

Everything here is GUI-independent and pure: callers hand in snapshots and
log entries, and get back flags and proposals.  Applying a proposal always
remains an explicit user action in the GUI.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

from circuit_structure import map_parameter_name, parameter_name_mapping
from eis_services import anomaly_parameter_transform

# Default similarity features.  The list is a parameter everywhere so the
# feature set can change later without touching this module's logic.
DEFAULT_SIMILARITY_FEATURES = ("time", "voltage", "current", "cycle", "cycle mod")

# How each logical feature name resolves against a metadata dict.  Plain
# strings are looked up directly (case-insensitive); entries marked with a
# leading "~" are treated as regular expressions matched against metadata
# keys (used for the "Cycle mod N" family of columns).
FEATURE_ALIASES: dict[str, tuple[str, ...]] = {
    "time": ("time", "time/s", "time_s"),
    "voltage": ("ecell_v", "potential_v", "working electrode potential (v)"),
    "current": ("i_ma", "current_ma", "<i>/mA".casefold(), "i (ma)"),
    "cycle": ("cycle", "cycle number"),
    "cycle mod": ("~cycle mod(?:\\s+\\d+)?",),
}


@dataclass(frozen=True)
class SpectrumFitSnapshot:
    """One spectrum's fit state plus the metadata used for similarity."""

    spectrum_id: str
    dataset_id: str
    dataset_label: str
    cycle: int
    control: str
    circuit: str
    parameter_names: tuple[str, ...]
    fitted_parameters: tuple[float, ...] | None
    fit_ok: bool
    residual: float | None = None
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def identity(self) -> str:
        return f"{self.dataset_id}::cycle-{self.cycle}"


@dataclass(frozen=True)
class NeighborEstimate:
    """Expected parameter values from nearby fitted spectra."""

    values: dict[str, float]
    contributors: tuple[tuple[str, float], ...]  # (identity, distance)
    used_features: tuple[str, ...]
    warnings: tuple[str, ...]
    # Raw neighbor values mapped onto the target's parameter names, used to
    # scale the per-parameter robust z-score locally.
    neighbor_values: dict[str, tuple[float, ...]]


@dataclass(frozen=True)
class ParameterDeviation:
    name: str
    actual: float
    expected: float
    robust_z: float


@dataclass(frozen=True)
class BadFitFlag:
    snapshot: SpectrumFitSnapshot
    reasons: tuple[str, ...]
    parameters: tuple[ParameterDeviation, ...] = ()
    residual: float | None = None

    @property
    def max_abs_z(self) -> float:
        return max((abs(p.robust_z) for p in self.parameters), default=0.0)


@dataclass(frozen=True)
class CorrectionProposal:
    snapshot: SpectrumFitSnapshot
    flag: BadFitFlag
    changes: tuple[dict[str, object], ...]
    sources: tuple[dict[str, object], ...]
    distance: float | None
    feature_names: tuple[str, ...]

    @property
    def identity(self) -> str:
        return self.snapshot.identity


def resolve_feature(
    metadata: dict[str, object], cycle: int | None, name: str
) -> float | None:
    """Resolve one logical feature name to a float, or ``None``."""
    key = str(name).strip().casefold()
    if key == "cycle" and cycle is not None:
        return float(cycle)
    aliases = FEATURE_ALIASES.get(key)
    if aliases is None:
        # Unknown feature: try the raw name so custom columns still work.
        aliases = (key,)
    lowered = {str(k).casefold(): v for k, v in metadata.items()}
    for alias in aliases:
        if alias.startswith("~"):
            pattern = re.compile(alias[1:], flags=re.IGNORECASE)
            for meta_key, value in metadata.items():
                if pattern.fullmatch(str(meta_key)):
                    converted = _as_float(value)
                    if converted is not None:
                        return converted
        else:
            if alias in lowered:
                converted = _as_float(lowered[alias])
                if converted is not None:
                    return converted
    return None


def _as_float(value: object) -> float | None:
    try:
        result = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


def feature_vector(
    metadata: dict[str, object],
    cycle: int | None,
    feature_names: tuple[str, ...] | list[str],
) -> np.ndarray | None:
    values = [resolve_feature(metadata, cycle, name) for name in feature_names]
    if any(value is None for value in values):
        return None
    return np.asarray(values, dtype=float)


def effective_features(
    snapshots: list[SpectrumFitSnapshot],
    feature_names: tuple[str, ...] | list[str] = DEFAULT_SIMILARITY_FEATURES,
    *,
    min_fraction: float = 0.5,
) -> tuple[str, ...]:
    """Keep the features that resolve for enough snapshots to be usable.

    Makes the similarity robust to datasets that lack, e.g., a time column;
    the caller can change the preferred feature list without this logic
    changing.
    """
    if not snapshots:
        return tuple(feature_names)
    threshold = max(1, int(np.ceil(min_fraction * len(snapshots))))
    usable = []
    for name in feature_names:
        count = sum(
            1
            for snapshot in snapshots
            if resolve_feature(snapshot.metadata, snapshot.cycle, name) is not None
        )
        if count >= threshold:
            usable.append(name)
    return tuple(usable) if usable else tuple(feature_names[:1])


def expected_parameters(
    target: SpectrumFitSnapshot,
    candidates: list[SpectrumFitSnapshot],
    *,
    feature_names: tuple[str, ...] | list[str] = DEFAULT_SIMILARITY_FEATURES,
    max_neighbors: int = 5,
) -> NeighborEstimate:
    """Estimate expected parameter values from nearby fitted spectra.

    Mirrors :func:`eis_services.suggest_eec_parameters` (range-normalized
    feature distance, inverse-square weights, geometric mean for positive
    parameters) but with a configurable feature list and snapshot type.
    """
    feature_names = tuple(feature_names)
    warnings: list[str] = []
    usable: list[tuple[SpectrumFitSnapshot, dict[str, float], float]] = []
    target_names = tuple(target.parameter_names)
    for candidate in candidates:
        if candidate.identity == target.identity:
            continue
        if not candidate.fit_ok or candidate.fitted_parameters is None:
            continue
        mapping = parameter_name_mapping(candidate.circuit, target.circuit)
        if mapping is None:
            continue
        mapped: dict[str, float] = {}
        ok = True
        for name, value in zip(candidate.parameter_names, candidate.fitted_parameters):
            target_name = map_parameter_name(name, mapping)
            if target_name is None or target_name in mapped:
                ok = False
                break
            mapped[target_name] = float(value)
        if not ok or set(mapped) != set(target_names):
            continue
        if not np.isfinite(np.asarray([mapped[n] for n in target_names], dtype=float)).all():
            continue
        target_vector = feature_vector(target.metadata, target.cycle, feature_names)
        candidate_vector = feature_vector(candidate.metadata, candidate.cycle, feature_names)
        if target_vector is None or candidate_vector is None:
            continue
        usable.append((candidate, mapped, 0.0))  # distance filled below
    if not usable:
        return NeighborEstimate({}, (), (), tuple(warnings + ["no compatible fitted neighbors"]), {})
    # Range-normalize each feature over target + candidates.
    vectors = [feature_vector(target.metadata, target.cycle, feature_names)]
    vectors += [feature_vector(c.metadata, c.cycle, feature_names) for c, _m, _d in usable]
    matrix = np.asarray(vectors, dtype=float)
    scales = matrix.max(axis=0) - matrix.min(axis=0)
    active = np.nonzero(scales > 0)[0]
    if active.size == 0:
        return NeighborEstimate({}, (), (), tuple(warnings + ["all similarity features are constant"]), {})
    used_features = tuple(feature_names[i] for i in active)
    target_vector = matrix[0]
    scored: list[tuple[SpectrumFitSnapshot, dict[str, float], float]] = []
    for candidate, mapped, _d in usable:
        vector = feature_vector(candidate.metadata, candidate.cycle, feature_names)
        assert vector is not None
        distance = float(
            np.linalg.norm((vector[active] - target_vector[active]) / scales[active])
        )
        scored.append((candidate, mapped, distance))
    scored.sort(key=lambda item: (item[2], item[0].identity))
    scored = scored[: max(1, int(max_neighbors))]
    exact = [item for item in scored if item[2] <= np.finfo(float).eps]
    selected = exact or scored
    distances = np.asarray([item[2] for item in selected], dtype=float)
    weights = (
        np.ones(distances.size, dtype=float)
        if exact
        else 1.0 / np.maximum(distances, 1e-12) ** 2
    )
    weights /= weights.sum()
    values: dict[str, float] = {}
    for name in target_names:
        parameter_values = np.asarray([item[1][name] for item in selected], dtype=float)
        if np.all(parameter_values > 0):
            values[name] = float(np.exp(np.sum(weights * np.log(parameter_values))))
        else:
            values[name] = float(np.sum(weights * parameter_values))
    contributors = tuple((item[0].identity, item[2]) for item in selected)
    neighbor_values = {
        name: tuple(item[1][name] for item in selected) for name in target_names
    }
    return NeighborEstimate(
        values=values,
        contributors=contributors,
        used_features=used_features,
        warnings=tuple(warnings),
        neighbor_values=neighbor_values,
    )


def flag_bad_fits(
    snapshots: list[SpectrumFitSnapshot],
    *,
    z_threshold: float,
    feature_names: tuple[str, ...] | list[str] = DEFAULT_SIMILARITY_FEATURES,
    residual_threshold: float | None = None,
    max_neighbors: int = 5,
    min_neighbors: int = 3,
) -> list[BadFitFlag]:
    """Flag spectra whose fitted parameters break the local smooth trend.

    For every fitted spectrum the expected value of each parameter is
    estimated from its nearest neighbors in feature space (time, voltage,
    current, cycle, cycle mod by default).  The per-parameter residual in the
    anomaly-transformed space is turned into a robust z-score across the
    whole set; a spectrum is flagged when any parameter exceeds
    ``z_threshold``.  Spectra without a successful fit are flagged as
    ``fit_failed``, and a mean relative fit residual above
    ``residual_threshold`` (when given) adds a ``high_residual`` reason.
    """
    if not np.isfinite(z_threshold) or z_threshold <= 0:
        raise ValueError("z_threshold must be positive")
    fitted = [s for s in snapshots if s.fit_ok and s.fitted_parameters is not None]
    deviations_by_snapshot: dict[str, list[ParameterDeviation]] = {}
    for snapshot in fitted:
        estimate = expected_parameters(
            snapshot, fitted, feature_names=feature_names, max_neighbors=max_neighbors
        )
        if not estimate.values or len(estimate.contributors) < min_neighbors:
            continue
        deviations: list[ParameterDeviation] = []
        for index, name in enumerate(snapshot.parameter_names):
            actual_t = anomaly_parameter_transform(name, float(snapshot.fitted_parameters[index]))
            if actual_t is None:
                continue
            # Local robust statistics over the nearest neighbors' transformed
            # values: median as expected, 1.4826*MAD as spread.  The median
            # keeps a single planted outlier from masking its own neighbors.
            neighbor_raw = estimate.neighbor_values.get(name, ())
            neighbor_t = [
                value
                for value in (
                    anomaly_parameter_transform(name, v) for v in neighbor_raw
                )
                if value is not None
            ]
            if len(neighbor_t) < 2:
                continue
            median_t = float(np.median(neighbor_t))
            mad_t = float(np.median(np.abs(np.asarray(neighbor_t) - median_t)))
            # The floor keeps floating-point dust from flagging exactly
            # constant parameters as infinite outliers.
            denominator = max(1.4826 * mad_t, 1e-9 * max(1.0, abs(median_t)))
            z = float(actual_t - median_t) / denominator
            deviations.append(
                ParameterDeviation(
                    name=name,
                    actual=float(snapshot.fitted_parameters[index]),
                    expected=float(np.median(neighbor_raw)),
                    robust_z=float(z),
                )
            )
        deviations_by_snapshot[snapshot.identity] = deviations
    flags: list[BadFitFlag] = []
    for snapshot in snapshots:
        reasons: list[str] = []
        if not snapshot.fit_ok or snapshot.fitted_parameters is None:
            reasons.append("fit_failed")
        deviations = deviations_by_snapshot.get(snapshot.identity, [])
        if any(abs(d.robust_z) > z_threshold for d in deviations):
            reasons.append("parameter_outlier")
        if (
            residual_threshold is not None
            and np.isfinite(residual_threshold)
            and snapshot.residual is not None
            and snapshot.residual > residual_threshold
        ):
            reasons.append("high_residual")
        if reasons:
            flags.append(
                BadFitFlag(
                    snapshot=snapshot,
                    reasons=tuple(reasons),
                    parameters=tuple(sorted(deviations, key=lambda d: -abs(d.robust_z))),
                    residual=snapshot.residual,
                )
            )
    flags.sort(key=lambda f: (-f.max_abs_z, f.snapshot.identity))
    return flags


def _log_entry_features(entry: dict) -> tuple[dict[str, object], int | None]:
    metadata = entry.get("metadata")
    metadata = dict(metadata) if isinstance(metadata, dict) else {}
    cycle = entry.get("cycle")
    try:
        cycle_value = int(cycle) if cycle is not None else None
    except (TypeError, ValueError):
        cycle_value = None
    return metadata, cycle_value


def propose_corrections(
    flags: list[BadFitFlag],
    log_entries: list[dict],
    *,
    feature_names: tuple[str, ...] | list[str] = DEFAULT_SIMILARITY_FEATURES,
    max_sources: int = 1,
) -> list[CorrectionProposal]:
    """Propose parameter/circuit changes replayed from logged corrections.

    For each flagged spectrum, the nearest logged correction entries (in the
    configurable feature space, same-project preferred) with a structurally
    compatible circuit are selected.  Parameter changes are mapped onto the
    target circuit via :func:`circuit_structure.parameter_name_mapping`;
    point-mask and frequency-window changes are deliberately not replayed
    (masks are spectrum-specific and must never be force-matched).  Circuit
    changes are replayed as a model proposal.
    """
    from ml.user_actions import correction_entries

    feature_names = tuple(feature_names)
    usable_entries = [
        entry
        for entry in correction_entries(log_entries)
        if entry.get("circuit")
    ]
    proposals: list[CorrectionProposal] = []
    for flag in flags:
        target = flag.snapshot
        target_vector = feature_vector(target.metadata, target.cycle, feature_names)
        if target_vector is None:
            continue
        scored: list[tuple[dict, dict[str, str]]] = []
        entry_vectors: list[np.ndarray] = []
        for entry in usable_entries:
            mapping = parameter_name_mapping(str(entry.get("circuit")), target.circuit)
            if mapping is None:
                continue
            entry_vector = feature_vector(*_log_entry_features(entry), feature_names)
            if entry_vector is None:
                continue
            scored.append((entry, mapping))
            entry_vectors.append(entry_vector)
        if not scored:
            continue
        # Distances are range-normalized per feature over target + entries.
        matrix = np.asarray([target_vector, *entry_vectors], dtype=float)
        scales = matrix.max(axis=0) - matrix.min(axis=0)
        active = np.nonzero(scales > 0)[0]
        if active.size == 0:
            distances = [0.0 for _ in scored]
        else:
            distances = [
                float(
                    np.linalg.norm(
                        (vector[active] - target_vector[active]) / scales[active]
                    )
                )
                for vector in entry_vectors
            ]
        order = sorted(
            range(len(scored)),
            key=lambda i: (distances[i], str(scored[i][0].get("timestamp", ""))),
        )
        chosen = order[: max(1, int(max_sources))]
        changes: list[dict[str, object]] = []
        sources: list[dict[str, object]] = []
        for index in chosen:
            entry, mapping = scored[index]
            changes_dict = entry.get("changes")
            changes_dict = changes_dict if isinstance(changes_dict, dict) else {}
            for raw_change in changes_dict.get("parameters", []) or []:
                if not isinstance(raw_change, dict):
                    continue
                mapped_name = map_parameter_name(str(raw_change.get("name", "")), mapping)
                if mapped_name is None:
                    continue
                changes.append(
                    {
                        "name": mapped_name,
                        "field": str(raw_change.get("field", "initial")),
                        "old": raw_change.get("old"),
                        "new": raw_change.get("new"),
                    }
                )
            circuit_change = changes_dict.get("circuit")
            if isinstance(circuit_change, dict) and circuit_change.get("new"):
                changes.append(
                    {
                        "name": None,
                        "field": "circuit",
                        "old": target.circuit,
                        "new": circuit_change.get("new"),
                    }
                )
            sources.append(
                {
                    "spectrum_id": entry.get("spectrum_id"),
                    "dataset_label": entry.get("dataset_label"),
                    "cycle": entry.get("cycle"),
                    "timestamp": entry.get("timestamp"),
                    "applied_by": entry.get("applied_by", "user"),
                }
            )
        if not changes:
            continue
        proposals.append(
            CorrectionProposal(
                snapshot=target,
                flag=flag,
                changes=tuple(changes),
                sources=tuple(sources),
                distance=distances[chosen[0]] if chosen and distances else None,
                feature_names=feature_names,
            )
        )
    return proposals

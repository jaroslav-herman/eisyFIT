import unittest

from ml.correction_replay import (
    DEFAULT_SIMILARITY_FEATURES,
    BadFitFlag,
    CorrectionProposal,
    SpectrumFitSnapshot,
    expected_parameters,
    feature_vector,
    flag_bad_fits,
    propose_corrections,
    resolve_feature,
)

CIRCUIT = "R0-p(R1,CPE1)"
NAMES = ("R0", "R1", "CPE1_Q", "CPE1_alpha")


def _snapshot(cycle, r1, *, fit_ok=True, circuit=CIRCUIT, residual=None, dataset_id="ds1"):
    metadata = {
        "Time": cycle * 60.0,
        "time/s": cycle * 60.0,
        "Ecell_V": 0.5 + 0.01 * cycle,
        "I_mA": 10.0 + 0.1 * cycle,
        "Cycle mod 15": cycle % 15,
        "potential_v": 0.5 + 0.01 * cycle,
        "current_ma": 10.0 + 0.1 * cycle,
    }
    return SpectrumFitSnapshot(
        spectrum_id=f"eisfit-spectrum-v1:{dataset_id}-{cycle}",
        dataset_id=dataset_id,
        dataset_label="Cell",
        cycle=cycle,
        control="EIS",
        circuit=circuit,
        parameter_names=NAMES,
        fitted_parameters=(10.0, r1, 1e-4, 0.8) if fit_ok else None,
        fit_ok=fit_ok,
        residual=residual,
        metadata=metadata,
    )


def _trend_snapshots(count=10):
    return [_snapshot(cycle, r1=100.0 + 5.0 * cycle) for cycle in range(1, count + 1)]


def _log_entry(cycle, circuit, changes, *, dataset_label="Cell"):
    return {
        "timestamp": f"2026-01-01T00:{cycle:02d}:00+00:00",
        "applied_by": "user",
        "action": f"cycle {cycle}: parameters",
        "dataset_id": "ds1",
        "dataset_label": dataset_label,
        "control": "EIS",
        "spectrum_id": f"eisfit-spectrum-v1:ds1-{cycle}",
        "cycle": cycle,
        "circuit": circuit,
        "metadata": {
            "Time": cycle * 60.0,
            "Ecell_V": 0.5 + 0.01 * cycle,
            "I_mA": 10.0 + 0.1 * cycle,
            "Cycle mod 15": cycle % 15,
        },
        "changes": {
            "cycle": str(cycle),
            "parameters": changes,
            "circuit": None,
            "frequency_window": None,
            "points_excluded": [],
            "points_included": [],
            "fit_changed": False,
        },
        "residual_before": 0.2,
        "residual_after": 0.05,
    }


class FeatureTests(unittest.TestCase):
    def test_resolve_aliases(self):
        metadata = {"Time": 120.0, "Ecell_V": 0.7, "I_mA": 5.0, "Cycle mod 15": 3}
        self.assertEqual(resolve_feature(metadata, 4, "time"), 120.0)
        self.assertEqual(resolve_feature(metadata, 4, "voltage"), 0.7)
        self.assertEqual(resolve_feature(metadata, 4, "current"), 5.0)
        self.assertEqual(resolve_feature(metadata, 4, "cycle"), 4.0)
        self.assertEqual(resolve_feature(metadata, 4, "cycle mod"), 3.0)

    def test_unknown_feature_uses_raw_name(self):
        self.assertEqual(resolve_feature({"custom col": 2.5}, None, "custom col"), 2.5)
        self.assertIsNone(resolve_feature({}, None, "time"))

    def test_feature_vector_requires_all_features(self):
        metadata = {"Time": 1.0, "Ecell_V": 0.5, "I_mA": 1.0, "Cycle mod 15": 1}
        vector = feature_vector(metadata, 2, DEFAULT_SIMILARITY_FEATURES)
        self.assertIsNotNone(vector)
        self.assertEqual(vector.size, len(DEFAULT_SIMILARITY_FEATURES))
        self.assertIsNone(feature_vector({"Time": 1.0}, 2, DEFAULT_SIMILARITY_FEATURES))


class FlagTests(unittest.TestCase):
    def test_parameter_outlier_is_flagged(self):
        snapshots = _trend_snapshots(10)
        snapshots[4] = _snapshot(5, r1=100000.0)  # planted bad fit
        flags = flag_bad_fits(snapshots, z_threshold=4.0)
        self.assertEqual(len(flags), 1)
        self.assertEqual(flags[0].snapshot.cycle, 5)
        self.assertIn("parameter_outlier", flags[0].reasons)
        self.assertEqual(flags[0].parameters[0].name, "R1")
        self.assertGreater(flags[0].max_abs_z, 4.0)

    def test_smooth_trend_not_flagged(self):
        snapshots = _trend_snapshots(10)
        flags = flag_bad_fits(snapshots, z_threshold=4.0)
        self.assertEqual(flags, [])

    def test_failed_fit_flagged(self):
        snapshots = _trend_snapshots(5) + [_snapshot(6, 0.0, fit_ok=False)]
        flags = flag_bad_fits(snapshots, z_threshold=4.0)
        failed = [f for f in flags if "fit_failed" in f.reasons]
        self.assertEqual(len(failed), 1)
        self.assertEqual(failed[0].snapshot.cycle, 6)

    def test_high_residual_reason(self):
        snapshots = _trend_snapshots(6)
        snapshots[2] = _snapshot(3, 115.0, residual=0.5)
        flags = flag_bad_fits(snapshots, z_threshold=4.0, residual_threshold=0.3)
        self.assertTrue(any("high_residual" in f.reasons for f in flags))

    def test_invalid_threshold_rejected(self):
        with self.assertRaises(ValueError):
            flag_bad_fits(_trend_snapshots(4), z_threshold=0.0)


class ExpectedParametersTests(unittest.TestCase):
    def test_nearest_neighbors_drive_estimate(self):
        snapshots = _trend_snapshots(10)
        target = snapshots[4]
        estimate = expected_parameters(target, snapshots)
        self.assertEqual(set(estimate.values), set(NAMES))
        self.assertEqual(len(estimate.contributors), 5)
        self.assertIn("time", estimate.used_features)
        # R1 trend is 100 + 5*cycle; the estimate at cycle 5 must be in range
        self.assertGreater(estimate.values["R1"], 100.0)
        self.assertLess(estimate.values["R1"], 200.0)
        self.assertIn("R1", estimate.neighbor_values)

    def test_configurable_feature_list(self):
        snapshots = _trend_snapshots(10)
        target = snapshots[4]
        estimate = expected_parameters(target, snapshots, feature_names=("cycle",))
        self.assertEqual(estimate.used_features, ("cycle",))
        self.assertEqual(len(estimate.contributors), 5)


class ProposalTests(unittest.TestCase):
    def test_proposal_replays_nearest_correction(self):
        snapshots = _trend_snapshots(10)
        snapshots[4] = _snapshot(5, r1=100000.0)
        flags = flag_bad_fits(snapshots, z_threshold=4.0)
        self.assertEqual(len(flags), 1)
        entry = _log_entry(
            4,
            CIRCUIT,
            [{"name": "R1", "field": "initial", "old": 120.0, "new": 250.0}],
        )
        proposals = propose_corrections(flags, [entry])
        self.assertEqual(len(proposals), 1)
        proposal = proposals[0]
        self.assertEqual(proposal.identity, flags[0].snapshot.identity)
        r1_changes = [c for c in proposal.changes if c["name"] == "R1"]
        self.assertEqual(len(r1_changes), 1)
        self.assertEqual(r1_changes[0]["new"], 250.0)
        self.assertEqual(r1_changes[0]["field"], "initial")
        self.assertEqual(len(proposal.sources), 1)
        self.assertEqual(proposal.sources[0]["cycle"], 4)

    def test_proposal_maps_across_equivalent_circuit(self):
        snapshots = _trend_snapshots(10)
        snapshots[4] = _snapshot(5, r1=100000.0)
        flags = flag_bad_fits(snapshots, z_threshold=4.0)
        # Logged correction used the reordered-but-equivalent circuit.
        entry = _log_entry(
            4,
            "R0-p(CPE1,R1)",
            [{"name": "R1", "field": "initial", "old": 120.0, "new": 250.0}],
        )
        proposals = propose_corrections(flags, [entry])
        self.assertEqual(len(proposals), 1)
        names = [c["name"] for c in proposals[0].changes]
        self.assertIn("R1", names)

    def test_point_and_window_changes_are_not_replayed(self):
        snapshots = _trend_snapshots(10)
        snapshots[4] = _snapshot(5, r1=100000.0)
        flags = flag_bad_fits(snapshots, z_threshold=4.0)
        entry = _log_entry(4, CIRCUIT, [])
        entry["changes"]["points_excluded"] = [0, 1, 2]
        entry["changes"]["frequency_window"] = {"old": [1.0, 1e5], "new": [10.0, 1e4]}
        proposals = propose_corrections(flags, [entry])
        # No parameter or circuit change -> no proposal at all.
        self.assertEqual(proposals, [])

    def test_incompatible_circuit_not_used(self):
        snapshots = _trend_snapshots(10)
        snapshots[4] = _snapshot(5, r1=100000.0)
        flags = flag_bad_fits(snapshots, z_threshold=4.0)
        entry = _log_entry(
            4,
            "R0-L0-p(R1,CPE1)-p(R2,CPE2)",
            [{"name": "R1", "field": "initial", "old": 120.0, "new": 250.0}],
        )
        proposals = propose_corrections(flags, [entry])
        self.assertEqual(proposals, [])

    def test_nearest_source_preferred(self):
        snapshots = _trend_snapshots(10)
        snapshots[4] = _snapshot(5, r1=100000.0)
        flags = flag_bad_fits(snapshots, z_threshold=4.0)
        near = _log_entry(4, CIRCUIT, [{"name": "R1", "field": "initial", "old": 1.0, "new": 111.0}])
        far = _log_entry(10, CIRCUIT, [{"name": "R1", "field": "initial", "old": 1.0, "new": 999.0}])
        proposals = propose_corrections(flags, [far, near])
        self.assertEqual(len(proposals), 1)
        self.assertEqual(proposals[0].changes[0]["new"], 111.0)
        self.assertEqual(proposals[0].sources[0]["cycle"], 4)


if __name__ == "__main__":
    unittest.main()

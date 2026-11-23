import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from ml.user_actions import (
    MAX_USER_ACTIONS,
    USER_ACTIONS_FORMAT,
    USER_ACTIONS_VERSION,
    append_user_actions,
    correction_entries,
    diff_cycle_dict,
    diff_parameters,
    diff_state_payloads,
    load_user_actions,
    load_user_actions_payload,
    make_action_entry,
    relative_fit_residual,
    user_actions_path_for,
    write_user_actions,
)


def _parameter(name, initial=1.0, lower=0.0, upper=100.0, fixed=False):
    return {
        "name": name,
        "unit": "Ohm",
        "initial": initial,
        "lower": lower,
        "upper": upper,
        "error_percent": None,
        "fixed": fixed,
    }


def _cycle_dict(**overrides):
    cycle = {
        "circuit": "R0-p(R1,CPE1)",
        "potential_v": 0.5,
        "current_ma": 1.0,
        "time_s": 60.0,
        "frequency_window": [1.0, 1e5],
        "auto_max_frequency": False,
        "manually_included": [True, True, True, False],
        "outliers": [False, False, False, True],
        "parameters": [_parameter("R0", 10.0), _parameter("R1", 50.0)],
        "fit_parameters": None,
        "fit_provenance": {},
    }
    cycle.update(overrides)
    return cycle


class UserActionsPathTests(unittest.TestCase):
    def test_plain_project(self):
        self.assertEqual(
            user_actions_path_for(Path("data/cell.eisfit.json")).name,
            "cell_user_actions.json",
        )

    def test_compressed_project(self):
        self.assertEqual(
            user_actions_path_for(Path("data/cell.eisfit.json.gz")).name,
            "cell_user_actions.json",
        )


class DiffTests(unittest.TestCase):
    def test_diff_parameters_captures_editable_fields_only(self):
        old = [_parameter("R0", 10.0), _parameter("R1", 50.0)]
        new = [_parameter("R0", 12.0, fixed=True), _parameter("R1", 50.0, upper=80.0)]
        changes = diff_parameters(old, new)
        by_field = {(c.name, c.field): c for c in changes}
        self.assertIn(("R0", "initial"), by_field)
        self.assertEqual(by_field[("R0", "initial")].new, 12.0)
        self.assertIn(("R0", "fixed"), by_field)
        self.assertIn(("R1", "upper"), by_field)
        self.assertNotIn(("R1", "initial"), by_field)
        # error_percent/unit are not editable fields and never diffed
        self.assertFalse(any(c.field in {"unit", "error_percent"} for c in changes))

    def test_diff_cycle_dict_points_and_fit(self):
        old = _cycle_dict()
        new = _cycle_dict(
            manually_included=[True, False, True, False],
            outliers=[False, True, False, True],
            fit_parameters=[10.0, 50.0],
        )
        change = diff_cycle_dict(old, new)
        self.assertIsNotNone(change)
        self.assertEqual(change.points_excluded, [1])
        self.assertEqual(change.fit_changed, True)

    def test_diff_cycle_dict_empty_when_unchanged(self):
        self.assertIsNone(diff_cycle_dict(_cycle_dict(), _cycle_dict()))

    def test_diff_cycle_dict_circuit_and_window(self):
        old = _cycle_dict()
        new = _cycle_dict(circuit="R0-p(R1,CPE1)-p(R2,CPE2)", frequency_window=[10.0, 1e4])
        change = diff_cycle_dict(old, new)
        self.assertEqual(change.circuit, ("R0-p(R1,CPE1)", "R0-p(R1,CPE1)-p(R2,CPE2)"))
        self.assertEqual(change.frequency_window, ([1.0, 1e5], [10.0, 1e4]))

    def test_diff_state_payloads_groups_by_dataset_and_cycle(self):
        old_payload = [
            {"dataset_id": "ds1", "state": {"cycles": {"1": _cycle_dict(), "2": _cycle_dict()}}, "dataframe": "a"},
            {"dataset_id": "ds2", "state": {"cycles": {"1": _cycle_dict()}}, "dataframe": "b"},
        ]
        new_payload = [
            {"dataset_id": "ds1", "state": {"cycles": {"1": _cycle_dict(parameters=[_parameter("R0", 99.0), _parameter("R1", 50.0)]), "2": _cycle_dict()}}, "dataframe": "a"},
            {"dataset_id": "ds2", "state": {"cycles": {"1": _cycle_dict()}}, "dataframe": "b"},
        ]
        results = diff_state_payloads(old_payload, new_payload)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["dataset_id"], "ds1")
        self.assertEqual(len(results[0]["cycles"]), 1)
        cycle_change = results[0]["cycles"][0]
        self.assertEqual(cycle_change["cycle"], "1")
        self.assertEqual(cycle_change["parameters"][0]["name"], "R0")
        self.assertEqual(cycle_change["parameters"][0]["new"], 99.0)


class SidecarIoTests(unittest.TestCase):
    def test_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "proj_user_actions.json"
            entry = make_action_entry(
                applied_by="user",
                action="cycle 3: parameters",
                dataset_id="ds1",
                dataset_label="Cell",
                control="EIS",
                spectrum_id="eisfit-spectrum-v1:abc",
                cycle=3,
                circuit="R0-p(R1,CPE1)",
                metadata={"potential_v": 0.5, "Time": 180},
                change={"cycle": "3", "parameters": [{"name": "R1", "field": "initial", "old": 50.0, "new": 80.0}], "circuit": None, "frequency_window": None, "points_excluded": [], "points_included": [], "fit_changed": False},
                residual_before=0.2,
                residual_after=0.05,
                timestamp="2026-01-01T00:00:00+00:00",
            )
            write_user_actions(path, [entry], source_project="proj.eisfit.json")
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(payload["format"], USER_ACTIONS_FORMAT)
            self.assertEqual(payload["version"], USER_ACTIONS_VERSION)
            entries = load_user_actions(path)
            self.assertEqual(len(entries), 1)
            self.assertEqual(entries[0]["spectrum_id"], "eisfit-spectrum-v1:abc")
            self.assertEqual(entries[0]["applied_by"], "user")

    def test_missing_file_returns_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(load_user_actions(Path(tmp) / "nope.json"), [])

    def test_foreign_format_rejected(self):
        with self.assertRaises(ValueError):
            load_user_actions_payload({"format": "something-else", "version": 1})

    def test_newer_version_rejected(self):
        with self.assertRaises(ValueError):
            load_user_actions_payload({"format": USER_ACTIONS_FORMAT, "version": USER_ACTIONS_VERSION + 1})

    def test_malformed_entries_dropped(self):
        payload = {
            "format": USER_ACTIONS_FORMAT,
            "version": USER_ACTIONS_VERSION,
            "entries": [{"cycle": 1}, "not-a-dict", 42, None],
        }
        entries = load_user_actions_payload(payload)
        self.assertEqual(entries, [{"cycle": 1}])

    def test_append_trims_to_limit(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "log.json"
            append_user_actions(path, [{"n": i} for i in range(MAX_USER_ACTIONS + 10)], limit=MAX_USER_ACTIONS)
            entries = load_user_actions(path)
            self.assertEqual(len(entries), MAX_USER_ACTIONS)
            self.assertEqual(entries[-1]["n"], MAX_USER_ACTIONS + 9)
            self.assertEqual(entries[0]["n"], 10)

    def test_correction_entries_filters(self):
        entries = [
            {"spectrum_id": "a", "changes": {"parameters": [{"name": "R1"}], "circuit": None, "points_excluded": []}},
            {"spectrum_id": "b", "changes": {"parameters": [], "circuit": None, "points_excluded": [3]}},
            {"spectrum_id": "c", "changes": {"parameters": [], "circuit": None, "points_excluded": []}},
            {"changes": {"parameters": [{"name": "R1"}]}},
            {"spectrum_id": "d", "changes": "not-a-dict"},
        ]
        selected = correction_entries(entries)
        self.assertEqual([e["spectrum_id"] for e in selected], ["a", "b"])


class ResidualTests(unittest.TestCase):
    def test_relative_fit_residual(self):
        frequency = np.array([1.0, 2.0, 3.0])
        z = np.array([1 + 1j, 2 + 2j, 3 + 3j])
        fit = np.array([1 + 1j, 2 + 2j, 3.3 + 3.3j])
        mask = np.array([True, True, True])
        residual = relative_fit_residual(frequency, z, fit, mask)
        # Only the third point deviates: |0.3+0.3j| / |3+3j| = 0.1, averaged
        # over the three included points.
        self.assertAlmostEqual(residual, 0.1 / 3)

    def test_relative_fit_residual_none_without_fit(self):
        self.assertIsNone(relative_fit_residual(np.ones(2), np.ones(2) * 1j, None, None))


if __name__ == "__main__":
    unittest.main()

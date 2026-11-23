"""Integration test: record a real state change through the payload diff path."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from eis_model import CycleState, ParameterValue, ProjectState
from eis_project import _dataframe_to_payload, _state_to_payload
from ml.results_schema import spectrum_identifier
from ml.user_actions import (
    append_user_actions,
    correction_entries,
    diff_state_payloads,
    load_user_actions,
    make_action_entry,
    relative_fit_residual,
    user_actions_path_for,
)


def _state(r0_initial=2.0) -> ProjectState:
    parameter = ParameterValue("R0", "Ohm", r0_initial, 0.0, 10.0, 1.5)
    cycle = CycleState(
        1,
        np.array([100.0, 10.0, 1.0]),
        np.array([2.0 - 0.1j, 3.0 - 0.2j, 4.0 - 0.3j]),
        manually_included=np.array([True, True, True]),
        parameters=[parameter],
        fit_parameters=np.array([r0_initial]),
        fit_at_data_impedance=np.array([2.1 - 0.1j, 3.1 - 0.2j, 4.1 - 0.3j]),
    )
    return ProjectState(
        Path("source.mpt"), "R0", "cell", [1], 1, [parameter], {1: cycle}
    )


def _payload(state) -> list:
    return [
        {
            "dataset_id": "ds",
            "state": _state_to_payload(state),
            "dataframe": "fingerprint",
        }
    ]


class RecordingIntegrationTests(unittest.TestCase):
    def test_parameter_edit_is_diffed_and_logged(self):
        old_state = _state(r0_initial=2.0)
        new_state = _state(r0_initial=5.0)
        old_payload = _payload(old_state)
        new_payload = _payload(new_state)
        diffs = diff_state_payloads(old_payload, new_payload)
        self.assertEqual(len(diffs), 1)
        self.assertEqual(diffs[0]["dataset_id"], "ds")
        cycle_change = diffs[0]["cycles"][0]
        self.assertEqual(cycle_change["parameters"][0]["name"], "R0")
        self.assertEqual(cycle_change["parameters"][0]["field"], "initial")
        self.assertEqual(cycle_change["parameters"][0]["old"], 2.0)
        self.assertEqual(cycle_change["parameters"][0]["new"], 5.0)

        cycle = new_state.cycles[1]
        entry = make_action_entry(
            applied_by="user",
            action="cycle 1: parameters",
            dataset_id="ds",
            dataset_label="Cell",
            control="cell",
            spectrum_id=spectrum_identifier(
                cycle.frequency_hz,
                cycle.impedance.real,
                cycle.impedance.imag,
                1,
                "cell",
            ),
            cycle=1,
            circuit=cycle.model(new_state.circuit),
            metadata={"potential_v": 0.0, "current_ma": 0.0, "cycle": 1},
            change=cycle_change,
            residual_before=relative_fit_residual(
                old_state.cycles[1].frequency_hz,
                old_state.cycles[1].impedance,
                old_state.cycles[1].fit_at_data_impedance,
                old_state.cycles[1].included,
            ),
            residual_after=relative_fit_residual(
                cycle.frequency_hz,
                cycle.impedance,
                cycle.fit_at_data_impedance,
                cycle.included,
            ),
        )
        with tempfile.TemporaryDirectory() as tmp:
            project = Path(tmp) / "cell.eisfit.json.gz"
            project.write_bytes(b"{}")  # only the sidecar path matters here
            sidecar = user_actions_path_for(project)
            self.assertEqual(sidecar.name, "cell_user_actions.json")
            append_user_actions(sidecar, [entry], source_project=str(project))
            entries = load_user_actions(sidecar)
            self.assertEqual(len(entries), 1)
            self.assertEqual(correction_entries(entries)[0]["changes"]["parameters"][0]["new"], 5.0)
            self.assertIsNotNone(entries[0]["residual_after"])
            self.assertTrue(entries[0]["spectrum_id"].startswith("eisfit-spectrum-v1:"))

    def test_no_change_no_diff(self):
        state = _state()
        self.assertEqual(diff_state_payloads(_payload(state), _payload(state)), [])

    def test_fit_only_change_marks_fit_changed_not_parameters(self):
        old_state = _state(r0_initial=2.0)
        new_state = _state(r0_initial=2.0)
        new_state.cycles[1].fit_provenance = {"method": "refined"}
        diffs = diff_state_payloads(_payload(old_state), _payload(new_state))
        self.assertEqual(len(diffs), 1)
        change = diffs[0]["cycles"][0]
        self.assertEqual(change["parameters"], [])
        self.assertTrue(change["fit_changed"])


if __name__ == "__main__":
    unittest.main()

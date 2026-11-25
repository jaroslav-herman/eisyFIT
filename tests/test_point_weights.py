import tempfile
import unittest
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from eis_model import CycleState, ParameterValue, ProjectState, copy_cycle_state
from eis_project import load_json_payload, load_project_file, save_project_file
from eis_services import (
    FitOptions,
    cycle_aligned_column,
    error_column_candidates,
    fit_cycle,
    load_cycle,
    weights_from_point_errors,
)
from impedance.models.circuits import CustomCircuit
from spectrum_simulator import simulate_spectrum

CIRCUIT = "R0-p(R1,CPE1)"
TRUE_VALUES = np.array([1.0, 2.0, 0.01, 0.9])


def _names(circuit, values):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = CustomCircuit(circuit, initial_guess=np.asarray(values, dtype=float))
    return model.get_param_names()[0]


def _parameters(circuit, values, scale=0.5):
    return [
        ParameterValue(name, "", scale * value, 0.0, 1e6)
        for name, value in zip(_names(circuit, values), values)
    ]


def _cycle(weights=None, frequency=None, impedance=None):
    if frequency is None:
        frequency = np.logspace(5, -2, 60)
    if impedance is None:
        impedance = simulate_spectrum(CIRCUIT, frequency, TRUE_VALUES).impedance
    parameters = _parameters(CIRCUIT, TRUE_VALUES)
    cycle = CycleState(1, frequency, impedance, parameters=parameters)
    if weights is not None:
        cycle.set_point_weights(weights)
    return cycle, parameters


def _manual_weighted_fit(frequency, impedance, weights, modulus=False):
    order = np.argsort(frequency)[::-1]
    frequency = np.asarray(frequency)[order]
    impedance = np.asarray(impedance)[order]
    weights = np.asarray(weights)[order]
    initial = np.array([0.5 * value for value in TRUE_VALUES])

    def residual(values):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = CustomCircuit(CIRCUIT, initial_guess=values)
            model.parameters_ = values
            difference = model.predict(frequency) - impedance
        if modulus:
            difference = difference / np.maximum(np.abs(impedance), np.finfo(float).eps)
        difference = difference * weights
        return np.concatenate((difference.real, difference.imag))

    result = least_squares(
        residual, initial, bounds=(np.zeros(4), np.full(4, 1e6)), max_nfev=100000
    )
    return result.x


class CycleWeightsModelTests(unittest.TestCase):
    def test_weights_default_to_one(self):
        cycle, _ = _cycle()
        np.testing.assert_array_equal(cycle.weights, np.ones(cycle.frequency_hz.size))
        self.assertFalse(cycle.has_custom_weights)

    def test_invalid_weights_are_rejected(self):
        cycle, _ = _cycle()
        with self.assertRaises(ValueError):
            cycle.set_point_weights(np.ones(5))
        with self.assertRaises(ValueError):
            cycle.set_point_weights(np.zeros(cycle.frequency_hz.size))
        with self.assertRaises(ValueError):
            cycle.set_point_weights(-np.ones(cycle.frequency_hz.size))
        nan_weights = np.ones(cycle.frequency_hz.size)
        nan_weights[3] = np.nan
        with self.assertRaises(ValueError):
            cycle.set_point_weights(nan_weights)
        with self.assertRaises(ValueError):
            CycleState(1, np.array([1.0, 2.0]), np.array([1.0j, 2.0j]), weights=[1.0, -1.0])

    def test_set_point_weight_clears_fit(self):
        cycle, _ = _cycle()
        cycle.fit_parameters = np.array([1.0, 2.0, 0.01, 0.9])
        cycle.set_point_weight(0, 5.0)
        self.assertEqual(cycle.weights[0], 5.0)
        self.assertIsNone(cycle.fit_parameters)
        self.assertTrue(cycle.has_custom_weights)
        cycle.reset_weights()
        np.testing.assert_array_equal(cycle.weights, np.ones(cycle.frequency_hz.size))
        self.assertFalse(cycle.has_custom_weights)

    def test_copy_cycle_state_carries_weights(self):
        cycle, _ = _cycle(weights=np.linspace(0.5, 2.0, 60))
        clone = copy_cycle_state(cycle)
        np.testing.assert_array_equal(clone.weights, cycle.weights)
        clone.weights[0] = 9.0
        self.assertNotEqual(clone.weights[0], cycle.weights[0])


class WeightedFitTests(unittest.TestCase):
    def test_general_path_matches_manual_weighted_least_squares(self):
        cycle, parameters = _cycle()
        weights = np.linspace(0.5, 3.0, cycle.frequency_hz.size)
        cycle.set_point_weights(weights)
        # Two least_squares stages force the general (non-curve_fit) path.
        options = FitOptions(pipeline=("least_squares", "least_squares"))
        result = fit_cycle(cycle, CIRCUIT, parameters, options)
        manual = _manual_weighted_fit(
            cycle.frequency_hz, cycle.impedance, weights
        )
        np.testing.assert_allclose(result.fitted_parameters, manual, rtol=1e-5, atol=1e-9)
        self.assertTrue(result.point_weight_summary["custom_weights"])
        self.assertEqual(
            result.point_weight_summary["weighted_points"], cycle.frequency_hz.size
        )

    def test_general_path_composes_with_modulus_weighting(self):
        cycle, parameters = _cycle()
        weights = np.linspace(0.5, 3.0, cycle.frequency_hz.size)
        cycle.set_point_weights(weights)
        options = FitOptions(
            pipeline=("least_squares", "least_squares"), weight_by_modulus=True
        )
        result = fit_cycle(cycle, CIRCUIT, parameters, options)
        manual = _manual_weighted_fit(
            cycle.frequency_hz, cycle.impedance, weights, modulus=True
        )
        np.testing.assert_allclose(result.fitted_parameters, manual, rtol=1e-5, atol=1e-9)

    def test_fast_path_weights_equivalent_to_modulus_weighting(self):
        # sigma = 1/w in the curve_fit path, so w = 1/|Z| must reproduce
        # weight_by_modulus exactly.
        cycle_modulus, parameters = _cycle()
        result_modulus = fit_cycle(
            cycle_modulus,
            CIRCUIT,
            parameters,
            FitOptions(weight_by_modulus=True),
        )
        cycle_weights, parameters = _cycle()
        cycle_weights.set_point_weights(
            1.0 / np.maximum(np.abs(cycle_weights.impedance), np.finfo(float).eps)
        )
        result_weights = fit_cycle(cycle_weights, CIRCUIT, parameters, FitOptions())
        np.testing.assert_allclose(
            result_weights.fitted_parameters,
            result_modulus.fitted_parameters,
            rtol=1e-6,
            atol=1e-10,
        )

    def test_weights_change_the_fit(self):
        rng = np.random.default_rng(3)
        frequency = np.logspace(5, -2, 60)
        impedance = simulate_spectrum(CIRCUIT, frequency, TRUE_VALUES).impedance
        impedance = impedance + rng.normal(0.0, 0.002, frequency.size) * (
            1.0 + 1.0j
        )
        cycle_plain, parameters = _cycle(frequency=frequency, impedance=impedance)
        plain = fit_cycle(cycle_plain, CIRCUIT, parameters, FitOptions())
        cycle_weighted, weighted_parameters = _cycle(
            frequency=frequency, impedance=impedance
        )
        weights = np.ones(frequency.size)
        weights[0] = 1000.0  # pull the fit toward the first point
        cycle_weighted.set_point_weights(weights)
        weighted = fit_cycle(cycle_weighted, CIRCUIT, weighted_parameters, FitOptions())
        self.assertFalse(
            np.allclose(weighted.fitted_parameters, plain.fitted_parameters, rtol=1e-4)
        )
        self.assertNotEqual(weighted.objective, plain.objective)

    def test_weights_stay_aligned_when_points_are_unsorted(self):
        frequency = np.logspace(5, -2, 60)
        impedance = simulate_spectrum(CIRCUIT, frequency, TRUE_VALUES).impedance
        weights = np.linspace(0.5, 3.0, frequency.size)
        rng = np.random.default_rng(7)
        permutation = rng.permutation(frequency.size)
        sorted_cycle, sorted_parameters = _cycle(
            weights=weights, frequency=frequency, impedance=impedance
        )
        sorted_result = fit_cycle(
            sorted_cycle,
            CIRCUIT,
            sorted_parameters,
            FitOptions(pipeline=("least_squares", "least_squares")),
        )
        shuffled_cycle, shuffled_parameters = _cycle(
            weights=weights[permutation],
            frequency=frequency[permutation],
            impedance=impedance[permutation],
        )
        shuffled_result = fit_cycle(
            shuffled_cycle,
            CIRCUIT,
            shuffled_parameters,
            FitOptions(pipeline=("least_squares", "least_squares")),
        )
        np.testing.assert_allclose(
            shuffled_result.fitted_parameters,
            sorted_result.fitted_parameters,
            rtol=1e-6,
            atol=1e-10,
        )

    def test_excluded_points_do_not_need_weights_but_keep_length(self):
        cycle, parameters = _cycle()
        cycle.manually_included[0] = False
        cycle.set_point_weights(np.full(cycle.frequency_hz.size, 2.0))
        result = fit_cycle(cycle, CIRCUIT, parameters, FitOptions())
        self.assertTrue(np.all(np.isfinite(result.fitted_parameters)))


class ErrorImportTests(unittest.TestCase):
    def test_weights_from_modulus_error(self):
        sigma = np.array([0.1, 0.5, 2.0])
        np.testing.assert_allclose(
            weights_from_point_errors(modulus_error=sigma), 1.0 / sigma
        )

    def test_weights_from_real_and_imaginary_errors(self):
        sigma_re = np.array([0.3, 0.0])
        sigma_im = np.array([0.4, 1.0])
        weights = weights_from_point_errors(real_error=sigma_re, imaginary_error=sigma_im)
        np.testing.assert_allclose(weights[0], 1.0 / 0.5)
        np.testing.assert_allclose(weights[1], 1.0)

    def test_invalid_errors_become_nan(self):
        weights = weights_from_point_errors(modulus_error=np.array([0.5, 0.0, -1.0, np.nan]))
        self.assertTrue(np.isfinite(weights[0]))
        self.assertTrue(np.all(~np.isfinite(weights[1:])))
        with self.assertRaises(ValueError):
            weights_from_point_errors()
        with self.assertRaises(ValueError):
            weights_from_point_errors(real_error=np.array([1.0]))

    def test_error_column_candidates_put_error_columns_first(self):
        columns = ["freq/Hz", "Re(Z)/Ohm", "err |Z|/Ohm", "sigma Re(Z)", "-Im(Z)/Ohm"]
        candidates = error_column_candidates(columns)
        self.assertEqual(candidates[:2], ["err |Z|/Ohm", "sigma Re(Z)"])
        self.assertEqual(sorted(candidates[2:]), sorted(columns[:1] + columns[1:2] + columns[4:]))

    def test_cycle_aligned_column_matches_load_cycle_order(self):
        rng = np.random.default_rng(11)
        frequency = rng.permutation(np.logspace(4, -1, 25))
        dataframe = pd.DataFrame(
            {
                "freq_hz": np.concatenate([frequency, np.logspace(3, 0, 10)]),
                "re_zwe_ce_ohm": np.arange(35, dtype=float),
                "minus_im_zwe_ce_ohm": np.arange(35, dtype=float) * 0.1,
                "ewe_ece_v": np.ones(35),
                "i_ma": np.ones(35),
                "cycle_number": [1] * 25 + [2] * 10,
                "err |Z|/Ohm": np.arange(35, dtype=float) + 1.0,
            }
        )
        cycle = load_cycle(dataframe, 1, "cell")
        errors = cycle_aligned_column(dataframe, 1, "err |Z|/Ohm")
        self.assertEqual(errors.size, cycle.frequency_hz.size)
        rows = dataframe["cycle_number"] == 1
        expected = dataframe.loc[rows, "err |Z|/Ohm"].to_numpy()[
            np.argsort(dataframe.loc[rows, "freq_hz"].to_numpy())[::-1]
        ]
        np.testing.assert_array_equal(errors, expected)
        # reference_frequency must match the loaded spectrum
        np.testing.assert_array_equal(
            cycle_aligned_column(
                dataframe, 1, "err |Z|/Ohm", reference_frequency=cycle.frequency_hz
            ),
            errors,
        )
        with self.assertRaises(ValueError):
            cycle_aligned_column(
                dataframe, 1, "err |Z|/Ohm", reference_frequency=cycle.frequency_hz[::-1]
            )
        with self.assertRaises(KeyError):
            cycle_aligned_column(dataframe, 1, "missing column")

    def test_imported_error_weights_flow_into_fit(self):
        frequency = np.logspace(5, -2, 60)
        impedance = simulate_spectrum(CIRCUIT, frequency, TRUE_VALUES).impedance
        sigma = np.linspace(0.001, 0.01, frequency.size)
        cycle, parameters = _cycle(frequency=frequency, impedance=impedance)
        cycle.set_point_weights(weights_from_point_errors(modulus_error=sigma))
        result = fit_cycle(
            cycle,
            CIRCUIT,
            parameters,
            FitOptions(pipeline=("least_squares", "least_squares")),
        )
        manual = _manual_weighted_fit(frequency, impedance, 1.0 / sigma)
        np.testing.assert_allclose(result.fitted_parameters, manual, rtol=1e-5, atol=1e-9)


class WeightPersistenceTests(unittest.TestCase):
    def _state_and_dataframe(self, weights=None):
        dataframe = pd.DataFrame(
            {
                "freq_hz": [100.0, 10.0, 1.0],
                "re_zwe_ce_ohm": [2.0, 3.0, 4.0],
                "minus_im_zwe_ce_ohm": [0.1, 0.2, 0.3],
                "ewe_ece_v": [1.0, 1.0, 1.0],
                "i_ma": [2.0, 2.0, 2.0],
                "cycle_number": [1, 1, 1],
            }
        )
        parameter = ParameterValue("R0", "Ohm", 2.0, 0.0, 10.0)
        cycle = CycleState(
            1,
            np.array([100.0, 10.0, 1.0]),
            np.array([2.0 - 0.1j, 3.0 - 0.2j, 4.0 - 0.3j]),
            parameters=[parameter],
        )
        if weights is not None:
            cycle.set_point_weights(weights)
        state = ProjectState(
            __import__("pathlib").Path("source.mpt"),
            "R0",
            "cell",
            [1],
            1,
            [parameter],
            {1: cycle},
        )
        return state, dataframe

    def test_weights_round_trip(self):
        weights = np.array([0.5, 1.0, 2.5])
        state, dataframe = self._state_and_dataframe(weights)
        with tempfile.TemporaryDirectory() as directory:
            path = __import__("pathlib").Path(directory) / "project.eisfit.json"
            save_project_file(state, path, datasets=[("source::cell", state, dataframe)])
            payload = load_json_payload(path)
            restored = load_project_file(state, dataframe, path, payload=payload)
            np.testing.assert_allclose(restored.cycles[1].weights, weights)

    def test_legacy_payload_without_weights_defaults_to_one(self):
        state, dataframe = self._state_and_dataframe()
        with tempfile.TemporaryDirectory() as directory:
            path = __import__("pathlib").Path(directory) / "project.eisfit.json"
            save_project_file(state, path, datasets=[("source::cell", state, dataframe)])
            payload = load_json_payload(path)
            del payload["cycles"]["1"]["weights"]
            restored = load_project_file(state, dataframe, path, payload=payload)
            np.testing.assert_array_equal(
                restored.cycles[1].weights, np.ones(3)
            )

    def test_incompatible_weight_array_is_rejected(self):
        state, dataframe = self._state_and_dataframe()
        with tempfile.TemporaryDirectory() as directory:
            path = __import__("pathlib").Path(directory) / "project.eisfit.json"
            save_project_file(state, path, datasets=[("source::cell", state, dataframe)])
            payload = load_json_payload(path)
            payload["cycles"]["1"]["weights"] = [1.0, 2.0]
            with self.assertRaises(ValueError):
                load_project_file(state, dataframe, path, payload=payload)


if __name__ == "__main__":
    unittest.main()

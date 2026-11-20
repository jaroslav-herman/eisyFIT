import unittest
import warnings

import numpy as np

from eis_model import CycleState, ParameterValue
from eis_services import (
    FitOptions,
    _compile_circuit_predict,
    _model_fit_curve,
    fit_cycle,
)
from impedance.models.circuits import CustomCircuit
from spectrum_simulator import simulate_spectrum


CIRCUITS = [
    ("R0-p(R1,CPE1)", [1.0, 2.0, 0.01, 0.9]),
    ("R0-L0-p(R1,CPE1)-p(R2,CPE2)", [5.0, 1e-7, 20.0, 1e-4, 0.8, 100.0, 1e-3, 0.7]),
    ("R0-p(R1,W1)-p(R2,CPE2)", [1.0, 2.0, 0.5, 3.0, 1e-4, 0.85]),
    ("R0-p(R1,G1)", [1.0, 2.0, 0.1, 0.01]),
    ("R0-p(R1,Gs1)", [1.0, 2.0, 0.1, 0.01, 0.5]),
    ("R0-p(R1,K1)", [1.0, 2.0, 0.1, 0.01]),
    ("R0-p(R1,Zarc1)", [1.0, 2.0, 0.1, 0.01, 0.8]),
    ("R0-p(R1,Wo1)", [1.0, 2.0, 0.1, 0.5]),
    ("R0-p(R1,Ws1)", [1.0, 2.0, 0.1, 0.5]),
    ("R0-p(R1,La1)", [1.0, 2.0, 1e-6, 0.9]),
    ("R0-C1-L1", [1.0, 1e-6, 1e-9]),
]


def _names(circuit, values):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = CustomCircuit(circuit, initial_guess=np.asarray(values, dtype=float))
    return model.get_param_names()[0]


class CompiledPredictTests(unittest.TestCase):
    def test_compiled_predict_matches_custom_circuit(self):
        frequency = np.logspace(5, -2, 60)
        for circuit, values in CIRCUITS:
            values = np.asarray(values, dtype=float)
            names = _names(circuit, values)
            compiled = _compile_circuit_predict(circuit, names, frequency, values)
            self.assertIsNotNone(compiled, circuit)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = CustomCircuit(circuit, initial_guess=values)
                model.parameters_ = values
                reference = model.predict(frequency)
            fast = compiled(frequency, values)
            np.testing.assert_allclose(fast, reference, rtol=1e-12, atol=1e-12)

    def test_unsupported_element_falls_back(self):
        frequency = np.logspace(4, -1, 10)
        values = np.array([1.0, 2.0, 0.1, 0.2, 0.3])
        names = _names("R0-p(R1,TLMQ1)", values)
        self.assertIsNone(_compile_circuit_predict("R0-p(R1,TLMQ1)", names, frequency, values))

    def test_model_fit_curve_matches_with_compiled_predict(self):
        circuit, values = CIRCUITS[1]
        values = np.asarray(values, dtype=float)
        names = _names(circuit, values)
        frequency = np.logspace(5, -2, 60)
        compiled = _compile_circuit_predict(circuit, names, frequency, values)
        f_fast, z_fast = _model_fit_curve(frequency, circuit, values, compiled_predict=compiled)
        f_ref, z_ref = _model_fit_curve(frequency, circuit, values)
        np.testing.assert_allclose(f_fast, f_ref, rtol=0, atol=0)
        np.testing.assert_allclose(z_fast, z_ref, rtol=1e-12, atol=1e-12)

    def test_default_fit_uses_fast_path_and_stays_accurate(self):
        circuit, true_values = CIRCUITS[1]
        frequency = np.logspace(5, -2, 60)
        impedance = simulate_spectrum(circuit, frequency, true_values).impedance
        names = _names(circuit, true_values)
        parameters = [
            ParameterValue(name, "", 0.5 * value, 0.0, 1e6)
            for name, value in zip(names, true_values)
        ]
        cycle = CycleState(1, frequency, impedance, parameters=parameters)
        result = fit_cycle(cycle, circuit, parameters, FitOptions())
        # The two p(R,CPE) branches are interchangeable, so allow their swap.
        swapped = np.array(true_values)[[0, 1, 5, 6, 7, 2, 3, 4]]
        if not np.allclose(result.fitted_parameters, true_values, rtol=1e-4, atol=1e-8):
            np.testing.assert_allclose(result.fitted_parameters, swapped, rtol=1e-4, atol=1e-8)
        self.assertTrue(result.converged)

    def test_fit_with_unsupported_element_still_works(self):
        circuit = "R0-p(R1,TLMQ1)"
        true_values = np.array([1.0, 2.0, 0.1, 0.2, 0.3])
        frequency = np.logspace(4, -1, 30)
        impedance = simulate_spectrum(circuit, frequency, true_values).impedance
        names = _names(circuit, true_values)
        parameters = [
            ParameterValue(name, "", 0.8 * value, 0.0, 1e6)
            for name, value in zip(names, true_values)
        ]
        cycle = CycleState(1, frequency, impedance, parameters=parameters)
        result = fit_cycle(cycle, circuit, parameters, FitOptions())
        self.assertTrue(np.all(np.isfinite(result.fitted_parameters)))


if __name__ == "__main__":
    unittest.main()

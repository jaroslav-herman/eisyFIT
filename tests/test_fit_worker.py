import os
import unittest

import numpy as np

import eis_services
from eis_model import CycleState, ParameterValue
from eis_services import FitOptions, FitTimeoutError, fit_cycle_with_timeout
from spectrum_simulator import simulate_spectrum


def _reset_fit_worker():
    """Terminate any cached fit worker so the next call spawns a cold one."""
    with eis_services._FIT_WORKER_LOCK:
        process = eis_services._FIT_WORKER_PROCESS
        connection = eis_services._FIT_WORKER_CONNECTION
        if process is not None and connection is not None:
            eis_services._shutdown_fit_worker(process, connection)
        elif connection is not None:
            connection.close()
            eis_services._FIT_WORKER_CONNECTION = None
        eis_services._FIT_WORKER_PROCESS = None


def _cycle(point_count=20):
    frequency = np.logspace(4, -1, point_count)
    impedance = simulate_spectrum(
        "R0-p(R1,CPE1)", frequency, [1.0, 2.0, 0.01, 0.9]
    ).impedance
    parameters = [
        ParameterValue("R0", "ohm", 1.0, 0.0, 10.0),
        ParameterValue("R1", "ohm", 2.0, 0.0, 10.0),
        ParameterValue("CPE1_0", "F", 0.01, 1e-6, 1.0),
        ParameterValue("CPE1_1", "", 0.9, 0.5, 1.0),
    ]
    return CycleState(1, frequency, impedance, parameters=parameters), parameters


class FitWorkerTests(unittest.TestCase):
    def setUp(self):
        _reset_fit_worker()

    def tearDown(self):
        _reset_fit_worker()

    def test_successful_fit_through_worker(self):
        cycle, parameters = _cycle()
        result = fit_cycle_with_timeout(
            cycle, "R0-p(R1,CPE1)", parameters, 120.0, FitOptions()
        )
        np.testing.assert_allclose(
            result.fitted_parameters, [1.0, 2.0, 0.01, 0.9], rtol=1e-6, atol=1e-9
        )
        self.assertEqual(result.fit_frequency_hz.size, 200)
        self.assertEqual(result.fit_impedance.size, 200)
        # The worker spawn must not leak BLAS thread overrides into the parent.
        self.assertNotIn("OPENBLAS_NUM_THREADS", os.environ)

    def test_worker_error_keeps_original_exception_type(self):
        cycle, parameters = _cycle(point_count=2)
        with self.assertRaises(ValueError) as context:
            fit_cycle_with_timeout(
                cycle, "R0-p(R1,CPE1)", parameters, 60.0, FitOptions()
            )
        self.assertIn("At least three included points", str(context.exception))

    def test_timeout_raises_fit_timeout_error_not_runtime_error(self):
        # A cold worker needs far longer than 0.05 s just to boot, so the
        # poll must time out deterministically.
        cycle, parameters = _cycle()
        with self.assertRaises(FitTimeoutError) as context:
            fit_cycle_with_timeout(
                cycle, "R0-p(R1,CPE1)", parameters, 0.05, FitOptions()
            )
        self.assertNotIsInstance(context.exception, RuntimeError)
        self.assertIn("0.05", str(context.exception))
        # The worker must be reusable after a timeout.
        result = fit_cycle_with_timeout(
            cycle, "R0-p(R1,CPE1)", parameters, 120.0, FitOptions()
        )
        np.testing.assert_allclose(
            result.fitted_parameters, [1.0, 2.0, 0.01, 0.9], rtol=1e-6, atol=1e-9
        )

    def test_invalid_timeout_rejected(self):
        cycle, parameters = _cycle()
        with self.assertRaises(ValueError):
            fit_cycle_with_timeout(cycle, "R0-p(R1,CPE1)", parameters, 0.0)


if __name__ == "__main__":
    unittest.main()

import importlib.util
import unittest
import numpy as np

from jssp_rl import (
    JSSPInstance,
    JobShopEnv,
    exact_serial_dispatch_oracle,
    generate_jssp_instance,
    solve_cpsat_exact,
)


@unittest.skipUnless(
    importlib.util.find_spec("gymnasium") is not None,
    "Gymnasium unavailable",
)
class GymnasiumIntegrationTests(unittest.TestCase):
    def test_environment_passes_gymnasium_checker(self):
        from gymnasium.utils.env_checker import check_env
        instance = generate_jssp_instance(seed=50, n_jobs=3, n_machines=3)
        env = JobShopEnv(instance)
        check_env(env, skip_render_check=True)


@unittest.skipUnless(
    importlib.util.find_spec("ortools") is not None,
    "OR-Tools unavailable",
)
class ORToolsIntegrationTests(unittest.TestCase):
    def test_cpsat_matches_independent_exhaustive_tiny_oracle(self):
        fixtures = [
            JSSPInstance(
                machines=np.array([[0, 1], [1, 0]], dtype=np.int64),
                durations=np.array([[3, 2], [2, 4]], dtype=np.int64),
            ),
            generate_jssp_instance(seed=51, n_jobs=3, n_machines=3),
        ]
        for instance in fixtures:
            exhaustive = exact_serial_dispatch_oracle(instance)
            cpsat = solve_cpsat_exact(instance, time_limit=10.0, workers=1)
            self.assertEqual(cpsat.status, "OPTIMAL")
            self.assertEqual(cpsat.makespan, exhaustive.makespan)


if __name__ == "__main__":
    unittest.main()

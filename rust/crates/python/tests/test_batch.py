import unittest

import numpy as np

from catanatron_rust import Batch


class BatchTests(unittest.TestCase):
    def test_observations_and_dynamic_menus_are_owned_and_versioned(self):
        batch = Batch(2, players=2, map="MINI", seed=7)
        observed = batch.observe_many([0, 1])

        self.assertEqual(observed["observation_schema_version"], 1)
        self.assertEqual(observed["action_schema_version"], 1)
        self.assertEqual(observed["features"].shape, (2, 194))
        self.assertEqual(observed["features"].dtype, np.int16)
        self.assertEqual(observed["menu_offsets"].shape, (3,))
        held = observed["features"].copy()

        first_id = int(observed["action_ids"][0])
        batch.step_many([0], [first_id])
        np.testing.assert_array_equal(observed["features"], held)

    def test_batch_validation_is_atomic_and_rejects_stale_ids(self):
        batch = Batch(2, players=2, seed=3)
        observed = batch.observe_many([0, 1])
        before = observed["features"].copy()
        first = int(observed["action_ids"][observed["menu_offsets"][0]])

        with self.assertRaisesRegex(ValueError, "stale"):
            batch.step_many([0, 1], [first, 0])
        np.testing.assert_array_equal(batch.observe_many([0, 1])["features"], before)

        with self.assertRaisesRegex(ValueError, "duplicate"):
            batch.reset_many([0, 0], [1, 2])
        np.testing.assert_array_equal(batch.observe_many([0, 1])["features"], before)

    def test_rollouts_preserve_roots_and_return_separate_truncation(self):
        batch = Batch(2, players=2, map="BASE")
        before = batch.observe_many([0, 1])["features"].copy()
        result = batch.rollout_many([0, 1], [10, 11], turn_limit=5, threads=2)

        self.assertEqual(result["rewards"].shape, (2, 2))
        self.assertEqual(result["truncated"].shape, (2,))
        np.testing.assert_array_equal(batch.observe_many([0, 1])["features"], before)


if __name__ == "__main__":
    unittest.main()

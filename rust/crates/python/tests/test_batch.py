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

    def test_rollout_batches_match_across_worker_counts(self):
        for size in (1, 16, 256):
            batch = Batch(size, players=2, map="MINI", seed=19)
            indices = list(range(size))
            seeds = [1000 + index for index in indices]
            expected = batch.rollout_many(indices, seeds, turn_limit=25, threads=1)
            for threads in (2, 4):
                actual = batch.rollout_many(indices, seeds, turn_limit=25, threads=threads)
                for field in ("winners", "rewards", "truncated"):
                    np.testing.assert_array_equal(actual[field], expected[field])

    def test_reset_is_explicit_and_validates_config(self):
        batch = Batch(2, players=2, map="MINI", seed=1)
        before = batch.observe_many([0, 1])["features"].copy()
        batch.reset_many([1], [99], config="MINI")
        after = batch.observe_many([0, 1])["features"]
        np.testing.assert_array_equal(after[0], before[0])
        with self.assertRaisesRegex(ValueError, "must match"):
            batch.reset_many([0], [3], config="BASE")

    def test_gym_catalogues_match_python_and_stable_ids_step(self):
        from catanatron.gym.envs.action_space import get_action_array
        from catanatron.models.enums import ActionType
        from catanatron.models.player import Color

        colors = (Color.RED, Color.BLUE, Color.ORANGE, Color.WHITE)
        for map_name in ("BASE", "MINI", "TOURNAMENT"):
            for players in (2, 3, 4):
                batch = Batch(1, players=players, map=map_name)
                observed = batch.observe_many([0])
                catalogue = get_action_array(colors[:players], map_name)
                self.assertEqual(
                    observed["gym_catalogue_size"],
                    len(catalogue),
                )
                self.assertEqual(
                    observed["gym_legal_mask"].shape,
                    (1, observed["gym_catalogue_size"]),
                )
                expected = [
                    index
                    for index, (action_type, _) in enumerate(catalogue)
                    if action_type == ActionType.BUILD_SETTLEMENT
                ]
                np.testing.assert_array_equal(
                    np.flatnonzero(observed["gym_legal_mask"][0]), expected
                )

        batch = Batch(1, players=2, map="MINI")
        observed = batch.observe_many([0])
        legal_id = int(np.flatnonzero(observed["gym_legal_mask"][0])[0])
        batch.step_gym_many([0], [legal_id])
        after = batch.observe_many([0])["features"].copy()
        with self.assertRaisesRegex(ValueError, "not legal"):
            batch.step_gym_many([0], [legal_id])
        np.testing.assert_array_equal(batch.observe_many([0])["features"], after)


if __name__ == "__main__":
    unittest.main()

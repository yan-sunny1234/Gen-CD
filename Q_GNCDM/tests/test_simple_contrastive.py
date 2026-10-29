import unittest

import numpy as np
import torch

from Q_GNCDM.simple_contrastive import (
    flip_observed_responses,
    make_student_episode,
    simple_infonce,
)


class SimpleContrastiveTests(unittest.TestCase):
    def test_flip_changes_only_observed_entries(self):
        x = np.array([[1, -1, 0, 1, -1], [0, 0, 1, 0, 0]], dtype=np.float32)
        noisy = flip_observed_responses(x, 0.1, np.random.default_rng(7))
        np.testing.assert_array_equal(noisy == 0, x == 0)
        self.assertEqual(int(np.sum(noisy[0] != x[0])), 1)
        self.assertEqual(int(np.sum(noisy[1] != x[1])), 1)
        np.testing.assert_array_equal(noisy[noisy != x], -x[noisy != x])

    def test_flip_does_not_modify_source(self):
        x = np.array([[1, -1, 0]], dtype=np.float32)
        original = x.copy()
        flip_observed_responses(x, 0.5, np.random.default_rng(8))
        np.testing.assert_array_equal(x, original)

    def test_episode_hides_queries_from_both_axes(self):
        train = np.array([[1, -1, 1], [-1, 1, -1]], dtype=np.float32)
        episode = make_student_episode(train, [0, 1], 0.34, np.random.default_rng(9))
        global_students = episode.student_ids[episode.query_students]
        self.assertTrue(np.all(episode.support[global_students, episode.query_items] == 0))
        expected = (train[global_students, episode.query_items] + 1.0) / 2.0
        np.testing.assert_array_equal(episode.query_labels, expected)

    def test_aligned_pairs_have_lower_loss_than_swapped_pairs(self):
        clean = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        aligned = simple_infonce(clean, clean, 0.2)
        swapped = simple_infonce(clean, clean.flip(0), 0.2)
        self.assertLess(float(aligned), float(swapped))

    def test_infonce_has_finite_gradients(self):
        clean = torch.randn(8, 4, requires_grad=True)
        noisy = torch.randn(8, 4, requires_grad=True)
        loss = simple_infonce(clean, noisy)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(clean.grad).all())
        self.assertTrue(torch.isfinite(noisy.grad).all())


if __name__ == "__main__":
    unittest.main()

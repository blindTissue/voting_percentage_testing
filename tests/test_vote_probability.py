import unittest

import numpy as np
from scipy.special import gammaln

from vote_probability import (
    alpha_from_counts,
    alpha_from_mean,
    collision_probability_dirichlet_multinomial,
    collision_probability_dirichlet_mixture,
    collision_probability_multinomial,
)


def _log_comb(n, k):
    return gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1)


def _direct_multinomial_collision(n_a, n_b, p_a, p_b):
    total = 0.0
    for x in range(min(n_a, n_b) + 1):
        for y in range(min(n_a - x, n_b - x) + 1):
            z_a = n_a - x - y
            z_b = n_b - x - y
            log_p_a = (
                gammaln(n_a + 1)
                - gammaln(x + 1)
                - gammaln(y + 1)
                - gammaln(z_a + 1)
                + x * np.log(p_a[0])
                + y * np.log(p_a[1])
                + z_a * np.log(p_a[2])
            )
            log_p_b = (
                gammaln(n_b + 1)
                - gammaln(x + 1)
                - gammaln(y + 1)
                - gammaln(z_b + 1)
                + x * np.log(p_b[0])
                + y * np.log(p_b[1])
                + z_b * np.log(p_b[2])
            )
            total += np.exp(log_p_a + log_p_b)
    return total


def _direct_dm_collision(n_a, n_b, alpha_a, alpha_b):
    def log_pmf(x, y, n, alpha):
        z = n - x - y
        alpha0 = np.sum(alpha)
        return (
            gammaln(n + 1)
            - gammaln(x + 1)
            - gammaln(y + 1)
            - gammaln(z + 1)
            + gammaln(alpha0)
            - gammaln(n + alpha0)
            + gammaln(x + alpha[0])
            - gammaln(alpha[0])
            + gammaln(y + alpha[1])
            - gammaln(alpha[1])
            + gammaln(z + alpha[2])
            - gammaln(alpha[2])
        )

    total = 0.0
    for x in range(min(n_a, n_b) + 1):
        for y in range(min(n_a - x, n_b - x) + 1):
            total += np.exp(log_pmf(x, y, n_a, alpha_a) + log_pmf(x, y, n_b, alpha_b))
    return total


class VoteProbabilityTest(unittest.TestCase):
    def test_alpha_from_counts_uses_counts_as_direction_and_kappa_as_strength(self):
        alpha = alpha_from_counts([3030, 1440, 78], kappa=1_000)

        np.testing.assert_allclose(alpha.sum(), 1_000)
        np.testing.assert_allclose(alpha, 1_000 * np.array([3030, 1440, 78]) / 4548)

    def test_alpha_from_mean_uses_u1_u2_and_implied_other_share(self):
        alpha = alpha_from_mean(u1=0.60, u2=0.30, kappa=50)

        np.testing.assert_allclose(alpha, np.array([30.0, 15.0, 5.0]))

    def test_alpha_from_mean_rejects_invalid_mean_share(self):
        with self.assertRaises(ValueError):
            alpha_from_mean(u1=0.70, u2=0.30, kappa=50)

    def test_alpha_from_mean_rejects_non_positive_kappa(self):
        with self.assertRaises(ValueError):
            alpha_from_mean(u1=0.60, u2=0.30, kappa=0)

    def test_multinomial_collision_matches_direct_enumeration_for_small_totals(self):
        p_a = np.array([0.55, 0.35, 0.10])
        p_b = np.array([0.50, 0.30, 0.20])

        expected = _direct_multinomial_collision(7, 6, p_a, p_b)
        actual = collision_probability_multinomial(7, 6, p_a, p_b)

        self.assertAlmostEqual(actual, expected, places=14)

    def test_dirichlet_multinomial_collision_matches_direct_enumeration_for_small_totals(self):
        alpha_a = np.array([8.0, 5.0, 2.0])
        alpha_b = np.array([7.0, 4.0, 3.0])

        expected = _direct_dm_collision(7, 6, alpha_a, alpha_b)
        actual = collision_probability_dirichlet_multinomial(7, 6, alpha_a, alpha_b)

        self.assertAlmostEqual(actual, expected, places=14)

    def test_one_component_mixture_equals_single_dirichlet_multinomial(self):
        alpha_a = np.array([8.0, 5.0, 2.0])
        alpha_b = np.array([7.0, 4.0, 3.0])

        expected = collision_probability_dirichlet_multinomial(7, 6, alpha_a, alpha_b)
        actual = collision_probability_dirichlet_mixture(7, 6, [1.0], [alpha_a], [1.0], [alpha_b])

        self.assertAlmostEqual(actual, expected, places=14)

    def test_mixture_collision_is_weighted_sum_of_component_pairs(self):
        alphas_a = [np.array([8.0, 5.0, 2.0]), np.array([3.0, 9.0, 2.0])]
        alphas_b = [np.array([7.0, 4.0, 3.0]), np.array([2.0, 4.0, 8.0])]
        weights_a = [0.25, 0.75]
        weights_b = [0.60, 0.40]

        expected = 0.0
        for weight_a, alpha_a in zip(weights_a, alphas_a):
            for weight_b, alpha_b in zip(weights_b, alphas_b):
                expected += weight_a * weight_b * collision_probability_dirichlet_multinomial(
                    7, 6, alpha_a, alpha_b
                )

        actual = collision_probability_dirichlet_mixture(7, 6, weights_a, alphas_a, weights_b, alphas_b)

        self.assertAlmostEqual(actual, expected, places=14)


if __name__ == "__main__":
    unittest.main()

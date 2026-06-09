import math
import unittest

from coin_match_probability import matching_head_count_probability


class CoinMatchProbabilityTest(unittest.TestCase):
    def test_one_flip_trials_match_half_the_time(self):
        self.assertAlmostEqual(matching_head_count_probability(1), 0.5)

    def test_two_flip_trials_match_three_eighths_of_the_time(self):
        self.assertAlmostEqual(matching_head_count_probability(2), 3 / 8)

    def test_probability_uses_central_binomial_identity(self):
        n = 10
        expected = math.comb(2 * n, n) / (4**n)

        self.assertAlmostEqual(matching_head_count_probability(n), expected)


if __name__ == "__main__":
    unittest.main()

import unittest
import numpy as np
from episodicdataset import combined_sdev_mean, get_combined_norm_stats

class TestCombinedStats(unittest.TestCase):

    def test_combined_sdev_mean(self):
        """
        Tests the combined_sdev_mean function by creating two datasets, combining them,
        and asserting that the function's output matches the stats of the combined dataset.
        """
        # Create two datasets d1 and d2
        d1 = np.random.rand(100, 5)
        d2 = np.random.rand(150, 5)

        # Create d3 as the combination of d1 and d2
        d3 = np.concatenate((d1, d2), axis=0)

        # Get sizes
        n1 = d1.shape[0]
        n2 = d2.shape[0]
        n3 = d3.shape[0]

        # Calculate means
        mean1 = np.mean(d1, axis=0)
        mean2 = np.mean(d2, axis=0)
        mean3 = np.mean(d3, axis=0)

        # Calculate sample standard deviations (ddof=1)
        std1 = np.std(d1, axis=0, ddof=1)
        std2 = np.std(d2, axis=0, ddof=1)
        std3 = np.std(d3, axis=0, ddof=1)

        # Calculate combined stats using the function
        combined_std, combined_mean = combined_sdev_mean(n1, mean1, std1, n2, mean2, std2)

        # Assert that the calculated combined stats are close to the actual stats of d3
        self.assertEqual(n3, n1 + n2)
        np.testing.assert_allclose(combined_mean, mean3, rtol=1e-2)
        np.testing.assert_allclose(combined_std, std3, rtol=1e-2)


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
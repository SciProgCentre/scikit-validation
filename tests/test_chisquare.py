import skvalidation
import numpy as np
import unittest

class PearsonChiSquaredExampleTest(skvalidation.PearsonChiSquaredTest):
    @unittest.expectedFailure
    def test_assertObservationByFrequencies(self):
        obs = np.array([16, 18, 16, 14, 12, 12])
        freq = np.array([16, 16, 16, 16, 16, 8])
        self.assertObservationByFrequencies(obs, freq)

    @unittest.expectedFailure
    def test_assertObservationByProbability(self):
        obs = np.array([16, 18, 16, 14, 12, 12])
        n = obs.sum()
        freq = np.array([16, 16, 16, 16, 16, 8])
        probability = freq/freq.sum()
        self.assertObservationByProbability(n, obs, probability)


if __name__ == '__main__':
    unittest.main()





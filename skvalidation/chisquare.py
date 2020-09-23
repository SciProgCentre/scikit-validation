import unittest
import numpy as np
from typing import Optional
from scipy.stats import chisquare


class PearsonChiSquaredTest(unittest.TestCase):

    def assertObservationByFrequencies(self, observation: np.ndarray, frequencies: np.ndarray,
                                       p_value: float = 0.95, degrees_of_freedom : int = 0, msg: Optional[str] = None):
        chi2, p_value_real = chisquare(observation, frequencies, ddof=degrees_of_freedom)
        if p_value_real < p_value:
            standard_msg = "Chi-squre test failed with chi2 = {}, p-value = {}, necessary p-value = {}".format(chi2, p_value_real, p_value)
            msg = self._formatMessage(msg, standard_msg)
            raise self.failureException(msg)

    def assertObservationByProbability(self, n: int, observation: np.ndarray, probability: np.ndarray,
                p_value: float = 5.0, degrees_of_freedom : int = 0, msg: Optional[str] = None):
        self.assertObservationByFrequencies(observation, n * probability, p_value, degrees_of_freedom, msg)



import unittest
import numpy as np
from typing import Optional
from scipy.stats import chisquare


class PearsonChiSquaredTest(unittest.TestCase):

    def assertObservationByFrequencies(self, observation: np.ndarray, frequencies: np.ndarray,
                                       p_value: float = 0.95, delta_degrees_of_freedom : int = 0, msg: Optional[str] = None):
        """The chi-square test tests the null hypothesis that the categorical data has the given frequencies.
         Fail if the calculated a p-value less that given *p-value*.

        :param observation: Observed frequencies in each category.
        :param frequencies: Expected frequencies in each category.
        :param p_value: minimal necessary p-value
        :param delta_degrees_of_freedom: The reduction in degrees of freedom
        """
        chi2, p_value_real = chisquare(observation, frequencies, ddof=delta_degrees_of_freedom)
        if p_value_real < p_value:
            standard_msg = "Chi-squre test failed with chi2 = {}, p-value = {}, necessary p-value = {}".format(chi2, p_value_real, p_value)
            msg = self._formatMessage(msg, standard_msg)
            raise self.failureException(msg)

    def assertObservationByProbability(self, n: int, observation: np.ndarray, probabilities: np.ndarray,
                                       p_value: float =  0.95, delta_degrees_of_freedom : int = 0, msg: Optional[str] = None):
        """The chi-square test tests the null hypothesis that the categorical data has the given probabilities.
         Fail if the calculated a p-value less that given *p-value*.

        :param n:
        :param observation: Observed frequencies in each category.
        :param probabilities: Expected probabilities in each category.
        :param p_value: minimal necessary p-value
        :param delta_degrees_of_freedom: The reduction in degrees of freedom
        """
        self.assertObservationByFrequencies(observation, n * probabilities, p_value, delta_degrees_of_freedom, msg)



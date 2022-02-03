import math

import numpy as np

from utilities.register import Register
from utilities.statistical_measurements_manager import StatisticalMeasurementsManager


class PersistenceIntervalStatistics:
    def __init__(self, **kwargs):
        self.simplex_tree = kwargs.get("simplex_tree", None)
        self.pintervals = []
        self.dimension_info = {}
        self.exclude_utliers = kwargs.get("exclude_outliers", False)
        self.statistical_mngr = StatisticalMeasurementsManager(self.get_lifetimes())

    def sanitize_intervals(self):
        try:
            if len(self.pintervals) == 0:
                raise Exception("ERROR: empty persistence interval collection!!")
            temp_hd = np.nan_to_num(self.pintervals, posinf=0, nan=0)  # we detect the max value if we need
            max_eps = temp_hd.max()
            del temp_hd

            temp2 = np.nan_to_num(self.pintervals, posinf=max_eps*1.25)  # replace inf values with max_eps + max_eps/4 = max_eps*1.25
            del self.pintervals
            self.pintervals = temp2
        except Exception as e:
            Register.add_error_message(e)
            raise e

    def get_persistence_intervals(self):
        """
        Detects the persistence intervals in the nearest dimension
        to the simplicial complex dimension. According to the following procedure:

        1. We get the filtered simplicial complex dimension (sc_dimension)
        2. We find a non-empty collection of persistence intervals whith dimension higher than 0
         in [0, sc_dimension).
        2.1. We decrement sc_dimension <-- sc_dimension - 1
        2.2. pintervals <-- []
        2.3 WHILE sc_dimension > 0:
        2.4    pintervals <-- get_intervals_from_complex(sc_dimension)
        2.5    sc_dimension <-- sc_dimension-1
        2.6 END WHILE

        :return: a list of persistence intervals pintervals
        """
        try:
            if len(self.pintervals) > 0:
                return self.pintervals

            dimension = self.simplex_tree.dimension()
            # dimension -= 1
            pintervals = []
            self.dimension_info = {}

            for d in range(1, dimension+1):
                pis = self.simplex_tree.persistence_intervals_in_dimension(d)
                if len(pis) > 0:
                    self.dimension_info.update({d: (len(pintervals), len(pis))})
                    pintervals.extend(pis)

            self.pintervals = pintervals

            self.sanitize_intervals()
            del pintervals

            return self.pintervals
        except Exception as e:
            Register.add_error_message(e)
            raise e

    def lifetime(self, pinterval):
        """
        Compute the lifetime of a persistence interval in the way of death-birth

        :param pinterval: the persistence interval
        :return: the lifetime of the interval
        """
        try:
            if len(pinterval) < 2 or math.isinf(pinterval[0]):
                Register.add_error_message("ERROR: bad persistence interval without birth time")
                raise Exception("ERROR: bad persistence interval without birth time")

            if math.isinf(pinterval[1]):
                Register.add_error_message("ERROR: immortal persistence interval, with an infty dead time")
                raise Exception("ERROR: immortal persistence interval, with an infty dead time")

            return pinterval[1] - pinterval[0]
        except Exception as e:
            Register.add_error_message(e)
            raise e

    def get_lifetimes(self):
        try:
            pintervals = self.get_persistence_intervals()

            lifetimes = []
            if len(pintervals) == 0:
                Register.add_error_message("ERROR: there are no persistence intervals")
                raise Exception("ERROR: there are no persistence intervals")

            for pi in pintervals:
                lftime = self.lifetime(pi)
                if not math.isinf(lftime):
                    lifetimes.append(lftime)

            return lifetimes
        except Exception as e:
            Register.add_error_message(e)
            raise e

    def maximum_lifetime(self):
        """
        Compute the maximum lifetime.

        :return: The maximum lifetime.
        """

        return self.statistical_mngr.max()

    def minimum_lifetime(self):
        """
        Compute the minimum lifetime.

        :return: The minimum lifetime.
        """
        try:
            return self.statistical_mngr.min()
        except Exception as e:
            Register.add_error_message(e)
            raise e

    def average_lifetime(self):
        """
         Detects the average lifetime from a collection of persistence intervals D with
         the closest lifetime to the average.
         N = |D|
         Avg(D) = 1/|D| * sum(lifetime(d)); \forall d \in D

         :return: the average lifetime
         """
        try:
            return self.statistical_mngr.avg()
        except Exception as e:
            Register.add_error_message(e)
            raise e

    def harmonic_mean_lifetime(self):
        """
         Detects the harmonic mean lifetime from a collection of persistence intervals D with
         the closest lifetime to the average.
         N = |D|
         hmean(D) = |D|/sum(1/lifetime(d)); \forall d \in D

         :return: the harmonic mean lifetime
         """
        try:
            return self.statistical_mngr.harmonic_mean()
        except Exception as e:
            Register.add_error_message(e)
            raise e

    def geometric_mean_lifetime(self):
        """
         Detects the geometric mean lifetime from a collection of persistence intervals D with
         the closest lifetime to the geometric mean.
         N = |D|
         Avg(D) = mult(lifetime(d)) ** (1/|D|); \forall d \in D

         :return: the geometric mean lifetime
         """
        try:
            return self.statistical_mngr.geometric_mean()
        except Exception as e:
            Register.add_error_message(e)
            raise e

    def median_lifetimes(self):
        """
        Compute the median of lifetimes from persistence intervals
        We sort persistent intervals according to its lifetimes
        and select the central position.

        :return: median of lifetimes
        """
        try:
            return self.statistical_mngr.median()
        except Exception as e:
            Register.add_error_message(e)
            raise e

    def std_from_lifetimes(self):
        """
        Compute the standard deviation of lifetimes from persistence intervals

        :return: stdev of lifetimes
        """
        try:
            return self.statistical_mngr.stdev()
        except Exception as e:
            Register.add_error_message(e)
            raise e

    def variance_from_lifetimes(self):
        """
        Compute the variance of lifetimes
        :return: variance of lifetimes
        """
        try:
            return self.statistical_mngr.variance()
        except Exception as e:
            Register.add_error_message(e)
            raise e

    def variation_coeff_from_lifetimes(self):
        """
        Compute the variation coefficient of lifetimes

        This coefficient is an interpretation of the variability degree of the sample independently of variable scale.
        This presents problems because of the sensitivity associated to the variable origin. That means, to higher
        values of this coefficient higher the heterogeneity of the variable. Lower values of this coefficient means
        higer homogeneity of variable values. For instance, if vcoeff <= 30% the average does represents the data.

        In other words, this coefficient can be used to understand if the average timelife is a good choice
        to be representative of the persistent interval collection.

        :return: variation coeff
        """
        try:
            stdev = self.std_from_lifetimes()
            avg = self.average_lifetime()

            if avg in (None, 0.0) or math.isinf(avg):
                Register.add_error_message("ERROR: there are no average lifespan")
                raise Exception("ERROR: there are no average lifespan")

            return stdev / avg
        except Exception as e:
            Register.add_error_message(e)
            raise e

    def interquartil_range(self):
        try:

            return self.statistical_mngr.IQR()
        except Exception as e:
            Register.add_error_message(e)
            raise e

    def detect_outliers(self):
        """
        Outlier detection according Turkey's test
        :return:
        """
        try:
            outliers_timespan = self.statistical_mngr.detect_outliers()

            return outliers_timespan
        except Exception as e:
            Register.add_error_message(e)
            raise e

    def get_dimension(self, pi):
        try:
            if pi is None:
                return None
            pos = -1
            for idx, pint in enumerate(self.pintervals):
                if pint[0] == pi[0] and pint[1] == pi[1]:
                    pos = idx

            if pos == -1:
                return min(self.dimension_info)

            for dim in self.dimension_info:
                interv = self.dimension_info[dim]
                if interv[0] <= pos <= interv[1]:
                    return dim

            return max(self.dimension_info)
        except Exception as e:
            Register.add_error_message(e)
            raise e

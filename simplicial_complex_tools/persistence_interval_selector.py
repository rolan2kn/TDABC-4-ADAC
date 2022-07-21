import math
from utilities.common_generators import CommonDataGenerator
from simplicial_complex_tools.persistence_interval_statistics import PersistenceIntervalStatistics
from utilities.register import Register


class PersistenceIntervalSelector:
    def __init__(self):
        pass

    def execute(self):
        pass


class NaivePersistenceIntervalSelector(PersistenceIntervalSelector):
    def __init__(self, simplex_tree):
        self.pi_statistics = PersistenceIntervalStatistics(simplex_tree=simplex_tree)
        super(NaivePersistenceIntervalSelector, self).__init__()

    def execute(self):
        pass

    def get_corresponding_dimension(self, pi):
        return self.pi_statistics.get_dimension(pi)

    def search_pi_by_lifetime(self, lifetime):
        try:
            if math.isinf(lifetime):
                raise Exception("ERROR: invalid lifetime, it cannot be infty")

            pintervals = self.get_persistence_intervals()
            if len(pintervals) == 0:
                raise Exception("ERROR: persistence intervals collection is empty")

            i = -1
            pi_size = len(pintervals)
            ltime = math.inf

            while i < pi_size and ltime != lifetime:
                i += 1
                ltime = self.pi_statistics.lifetime(pintervals[i])

            pi = None
            if ltime == lifetime:
                pi = pintervals[i]

            # del pintervals

            return pi
        except Exception as e:
            Register.add_error_message(e)
            raise e

    def search_closest_pi_to_lifetime(self, lifetime):
        try:
            if math.isinf(lifetime):
                raise Exception("ERROR: invalid lifetime, it cannot be infty")

            pintervals = self.get_persistence_intervals()
            if len(pintervals) == 0:
                raise Exception("ERROR: persistence intervals collection is empty")


            c_pi = pintervals[0]
            minimal_diff = math.fabs(self.pi_statistics.lifetime(c_pi) - lifetime)

            for idx, interv in enumerate(pintervals):
                ltime = self.pi_statistics.lifetime(interv)
                if not math.isinf(ltime):
                    current_diff = math.fabs(ltime - lifetime)
                    if current_diff <= minimal_diff:
                        minimal_diff = current_diff
                        c_pi = interv

            return c_pi
        except Exception as e:
            Register.add_error_message(e)
            raise e

    def get_persistence_intervals(self):
        """
        Interface to PersistenceIntervalStatistics.get_persistence_intervals()

        :return: the collection of persistence intervals
        """
        return self.pi_statistics.get_persistence_intervals()

    def max_lifetime_interval(self):
        """
        Detects the persistence interval with the maximum lifetime.

        :return: the persistence interval with maximal lifetime
        """
        max_lifetime = self.pi_statistics.maximum_lifetime()

        return self.search_pi_by_lifetime(max_lifetime)

    def min_lifetime_interval(self):
        """
        Detects the persistence interval with the minimum lifetime.

        :return: the persistence interval with minimum lifetime
        """
        min_lifetime = self.pi_statistics.minimum_lifetime()

        return self.search_pi_by_lifetime(min_lifetime)

    def closest_to_average_lifetime(self):
        """
        Detects the persistence interval with the closest lifetime to the average lifetime.

        :return: the selected persistence interval
        """
        avg_lifetime = self.pi_statistics.average_lifetime()

        return self.search_closest_pi_to_lifetime(avg_lifetime)

    def closest_to_median_lifetime(self):
        """
        Detects the persistence interval with the closest lifetime to the median lifetime.

        :return: the selected persistence interval
        """
        median_lifetime = self.pi_statistics.median_lifetimes()

        return self.search_closest_pi_to_lifetime(median_lifetime)

    def closest_to_harmonic_mean_lifetime(self):
        """
        Detects the persistence interval with the closest lifetime to the harmonic mean lifetime.

        :return: the selected persistence interval
        """
        hm_lifetime = self.pi_statistics.harmonic_mean_lifetime()

        return self.search_closest_pi_to_lifetime(hm_lifetime)

    def closest_to_geometric_mean_lifetime(self):
        """
        Detects the persistence interval with the closest lifetime to the geometric mean lifetime.

        :return: the selected persistence interval
        """
        gm_lifetime = self.pi_statistics.geometric_mean_lifetime()

        return self.search_closest_pi_to_lifetime(gm_lifetime)

    def lower_half_randomized_lifetime(self):
        """
        Select a randomized persistence interval between
        the minimum and the closest to average lifetimes

        :return: selected_persistence interval
        """

        min = self.pi_statistics.minimum_lifetime()
        avg = self.pi_statistics.average_lifetime()

        if math.isinf(min) or math.isinf(avg):
            return None

        selected_lifetime = CommonDataGenerator.random_float(min, avg)

        randomized_pi = self.search_closest_pi_to_lifetime(selected_lifetime)

        return randomized_pi

    def upper_half_randomized_lifetime(self):
        """
        Select a randomized persistence interval between
        the closes to average lifetime and the maximum lifetime

        :return: selected_persistence interval
        """

        max = self.pi_statistics.maximum_lifetime()
        avg = self.pi_statistics.average_lifetime()

        if math.isinf(max) or math.isinf(avg):
            return None

        selected_lifetime = CommonDataGenerator.random_float(avg, max)

        randomized_pi = self.search_closest_pi_to_lifetime(selected_lifetime)

        return randomized_pi

    def fully_randomized_lifetime(self):
        """
        Select a randomized persistence interval between
        the minimum and the maximum lifetimes

        :return: selected_persistence interval
        """

        max = self.pi_statistics.maximum_lifetime()
        min = self.pi_statistics.minimum_lifetime()

        if math.isinf(max) or math.isinf(min):
            return None

        selected_lifetime = CommonDataGenerator.random_float(min, max)

        randomized_pi = self.search_closest_pi_to_lifetime(selected_lifetime)

        return randomized_pi


class LocalHomologyFeatureSelector(PersistenceIntervalSelector):
    def __init__(self):
        pass

    def execute(self):
        pass


class CombinatorialLaplacianFeatureSelector(PersistenceIntervalSelector):
    def __init__(self):
        pass

    def execute(self):
        pass


class StatisticalFeatureSelector(PersistenceIntervalSelector):
    def __init__(self):
        pass

    def execute(self):
        pass

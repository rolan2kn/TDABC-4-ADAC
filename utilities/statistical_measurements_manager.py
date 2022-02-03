import numpy as np
from scipy.stats import hmean, gmean

"""
Wrapper class for Statistical Computations
"""


class StatisticalMeasurementsManager:
    def __init__(self, data):
        if len(data) == 0:
            raise Exception("StatisticalMeasurementsManager: Data esta Vacio")

        self.data = np.array(data)

    def mean(self):
        return np.mean(self.data)

    def harmonic_mean(self):
        return hmean(self.data)

    def geometric_mean(self):
        return gmean(self.data)

    def avg(self):
        return np.average(self.data)

    def max(self):
        return np.max(self.data)

    def min(self):
        return np.min(self.data)

    def median(self):
        return np.median(self.data)

    def stdev(self):
        return np.std(self.data)

    def variance(self):
        return np.var(self.data)

    def percentile(self, percentage):
        return np.percentile(self.data, percentage)

    def IQR(self):
        return self.percentile(75) - self.percentile(25)

    def detect_outliers(self):
        outlier_factor = 1.5*self.IQR()
        outliers = self.data[(self.data < self.percentile(25)-outlier_factor) | (self.data > self.percentile(75)+outlier_factor)]

        pos, = np.where(outliers)

        return self.data[pos]

    def z_score(self):
        pass

    def cohen_d(self):
        pass

    def p_value(self):
        pass
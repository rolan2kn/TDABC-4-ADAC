#!/usr/bin/python
# -*- coding: utf-8 -*-
import time

from dataset_tools.datasets_common_generator import DatasetCommonGenerator
from scipy.stats import truncnorm
import numpy as np
import random


class NormalDistBasedDatasetsGenerator(DatasetCommonGenerator):
    def __init__(self, *args, **kwargs):
        super(NormalDistBasedDatasetsGenerator, self).__init__(*args, **kwargs)
        self.is_uniform = kwargs.get("is_uniform", False)

    def make_multivariate2(self, mean, stdev, size, dim, idx=0):
        np.random.seed(idx)

        X = mean*np.ones((size, dim)) + stdev*np.random.randn(size, dim)
        self.X.extend(X)

    def make_multivariate(self, mean, stdev, size, dim, idx=0):
        values = []
        np.random.seed(idx)

        for d in range(dim):
            if not self.is_uniform:
                values.append(self.random_norm_dist(mean, stdev, size))
            else:
                values.append(self.random_uniform(mean, stdev, size))

        new_V = np.array(values).T
        self.X.extend(new_V)

        del values
        del new_V

    def random_norm_dist(self, mean, stdev, size):
        return np.random.normal(mean, 2*stdev, size)

    def random_uniform(self, mean, stdev, size):
        return np.random.uniform(mean, stdev, size)

    def get_truncated_normal(self, mean=0, sd=1, low=0, upp=10):
        return truncnorm(
            (low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)

    def configure_mean(self, no_object, mean=None):
        if mean is None:
            mean = [0]*no_object
        else:
            if type(mean) is not list:
                mean = [mean]
            m = len(mean)
            val = mean[0]
            if m < no_object:
                for i in range(m, no_object):
                    mean.append(val)
        return mean

    def configure_std(self, no_object, stdev=None):
            if stdev is None:
                stdev = [1] * no_object
            else:
                if type(stdev) is not list:
                    stdev = [stdev]
                s = len(stdev)
                val = stdev[0]
                if s < no_object:
                    for i in range(s, no_object):
                        stdev.append(val)
            return stdev


    def configure_arguments(self, **kwargs):
        kwargs = self.cure_attrib(**kwargs)

        no_object = kwargs.get("no_objects", None)
        if no_object is None:
            no_object = kwargs.get("no_object", 1)
            if no_object is None:
                no_object = 4
        sizes = self.configure_sizes(no_object, kwargs.get("sizes", None))
        mean = self.configure_mean(no_object, kwargs.get("mean", None))
        stdev = self.configure_std(no_object, kwargs.get("stdev", None))
        save = kwargs.get("save", False)
        scale = kwargs.get("scale", 1)

        str_size = "_".join([str(s) for s in sizes])
        self.suffix = "{0}_{6}_{4}m_{5}std_{1}s_{2}d_{3}".format(no_object, scale,
                                                                 self.dim,
                                                                 str_size,
                                                                 mean[0],
                                                                 stdev[0],
                                                                 self.get_class())

        return no_object, sizes, mean, stdev, scale, save

    def make_intersected(self, **kwargs):
        no_objects, sizes, mean, stdev, scale, save = self.configure_arguments(**kwargs)

        prev_size = len(self.X)
        self.make_multivariate(mean[0], stdev[0], sizes[0], self.dim, idx=0)
        current_size = len(self.X)
        if current_size-prev_size > 0:
            super(NormalDistBasedDatasetsGenerator, self).compute_labels(
                current_size - prev_size)

        for d in range(0, no_objects-1):
            prev_size = current_size
            self.make_multivariate(mean[d+1], stdev[d+1], sizes[d+1], self.dim, idx=d+1)
            current_size = len(self.X)
            if current_size - prev_size > 0:
                super(NormalDistBasedDatasetsGenerator, self).compute_labels(current_size-prev_size)

        self.samples_number = len(self.X)

        if save:
            self.save_dataset()

        return self.X
#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt

import math

from dataset_tools.datasets_common_generator import DatasetCommonGenerator


class SphereBasedDatasetGenerator(DatasetCommonGenerator):
    def __init__(self, *args, **kwargs):
        super(SphereBasedDatasetGenerator, self).__init__(*args, **kwargs)

    def abstract_generator(self, **kwargs):
        self.randsphere(kwargs.get("size", self.samples_number),
                        kwargs.get("translate", None),
                        kwargs.get("scale", None))

    def randsphere(self, n, translate = None, scale = 1.0):
        '''
        Helper function to make an array of random numbers having shape (n, )
        with each number distributed Uniform(vmin, vmax).
        '''

        np.random.seed(19680801)
        translate = [0,0,0] if translate is None else translate

        z = 2*np.random.rand(n) - 1  # uniform on [vmin, vmax]

        theta = 2 * math.pi * np.random.rand(n) - math.pi  # uniform on [-pi, pi]
        x = np.sin(theta) * np.sqrt(1 - z ** 2)  # based on angle
        y = np.cos(theta) * np.sqrt(1 - z ** 2)

        for i in range(n):
            self.X.extend([[(x[i]+translate[0])*scale, (y[i]+translate[1])*scale, (z[i]+translate[2])*scale]])
        return self.X

    def configure_arguments(self, **kwargs):
        kwargs = self.cure_attrib(**kwargs)
        no_object = kwargs.get("no_objects", None)
        if no_object is None:
            no_object = kwargs.get("no_object", 1)
        sizes = self.configure_sizes(no_object, kwargs.get("sizes", None))
        scale = kwargs.get("scale", 1)

        str_size = "_".join([str(s) for s in sizes])
        self.suffix = "{0}n_{4}_{1}s_{2}d_{3}".format(no_object, scale,
                                                                self.dim,
                                                                str_size, self.get_class())

        save = kwargs.get("save", True)

        return no_object, sizes, scale, save

    def make_intersected(self, **kwargs):
        no_spheres, sizes, scale, save = self.configure_arguments(**kwargs)

        self.randsphere(sizes[0], None, scale)
        self.compute_labels(sizes[0])
        for d in range(0, no_spheres-1):
            translate = self.compute_translation(d, scale)
            self.randsphere(sizes[d+1], translate, scale)
            self.compute_labels(sizes[d+1])

        self.samples_number = len(self.X)
        if save:
            self.save_dataset()

        return self.X


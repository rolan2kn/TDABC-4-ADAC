#!/usr/bin/python
# -*- coding: utf-8 -*-
import os.path

import h5py
from sklearn import datasets
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import make_swiss_roll

from dataset_tools.dataset_transformer import DatasetTransformer
from dataset_tools.normal_distribution_based_datasets_generator \
    import NormalDistBasedDatasetsGenerator
from dataset_tools.sphere_based_datasets_generator \
    import SphereBasedDatasetGenerator

from dataset_tools.dataset_splitter import DatasetSplitter
from dataset_tools.dataset_settings import DatasetSettingHandler
from utilities import utils
import numpy as np


class DatasetBuilder:
    def __init__(self, metadata=None):
        if metadata is None:
            metadata = DatasetSettingHandler()
        self.metadata = metadata
        self.metadata.execute()

    def execute(self, *args):
        if len(args) > 0:
            self.load_from_scratch(*args)

        elif self.metadata.dtype_hdler.is_iris():
            self.load_iris()

        elif self.metadata.dtype_hdler.is_swissroll():
            self.load_swiss_roll()

        elif self.metadata.dtype_hdler.is_moon():
            self.load_moons()

        elif self.metadata.dtype_hdler.is_circles():
            self.load_circles()

        elif self.metadata.dtype_hdler.is_sphere():
            self.load_spheres()

        elif self.metadata.dtype_hdler.is_normal_dist() or self.metadata.dtype_hdler.is_krnn():
            self.load_normal_dist()

        elif self.metadata.dtype_hdler.is_breast_cancer():
            self.load_breast_cancer()

        else:
            self.load_wine()

        return self.metadata

    def load_iris(self):
        iris = datasets.load_iris()
        self.get_common_data(iris.data, iris.target)

        self.metadata.labels = list(iris.target_names)

    def load_from_scratch(self, *args):
        """

        :param args: Should be a two element array, in position 0 X should arrive, and position 1 should hold the targets
        :return:
        """
        self.get_common_data(X=args[0], Y=args[1])

    def load_swiss_roll(self):
        n_samples = self.metadata.samples_count
        noise = self.metadata.noise

        X, _ = make_swiss_roll(n_samples=n_samples, noise=noise)
        ward = AgglomerativeClustering(n_clusters=6, linkage='ward').fit(X)

        self.get_common_data(X, ward.labels_)

    def get_common_data(self, X, Y, suffix=None, **kwargs):
        self.metadata.configure_dimensionality_reduction()

        dataset = DatasetTransformer(data_points=X, ttype=self.metadata.data_transformation_type, **self.metadata.args).execute()
        tags = Y
        if suffix is None:
            suffix = "{0}_{1}d_{2}noise_{3}samples".format(
                self.metadata.dataset_name,
                self.metadata.dimensions,
                self.metadata.noise,
                self.metadata.samples_count)

        self.metadata.splitter = DatasetSplitter(dataset=dataset, tags=tags)
        self.metadata.tags_set = set(tags)
        self.metadata.suffix = suffix

    def load_circles(self):
        noise = self.metadata.noise

        if self.metadata.samples_count is None:
            self.metadata.samples_count = 1500

        samples_count = self.metadata.samples_count

        X, Y = datasets.make_circles(n_samples=samples_count,
                                     noise=noise)

        self.get_common_data(X, Y)

    def load_moons(self):
        # generate 2d classification dataset
        if self.metadata.noise is None:
            self.metadata.noise = 0.1
        if self.metadata.samples_count is None:
            self.metadata.samples_count = 500

        noise = self.metadata.noise
        n_samples = self.metadata.samples_count

        X, Y = datasets.make_moons(n_samples=n_samples, noise=noise)

        self.get_common_data(X, Y)

    def load_spheres(self):
        if self.metadata.samples_count is None:
            self.metadata.samples_count = 100
        if self.metadata.dimensions is None:
            self.metadata.dimensions = 3

        samples_count = self.metadata.samples_count
        dim = self.metadata.dimensions
        stdev = self.metadata.stdev

        sphere_dataset = SphereBasedDatasetGenerator(
            samples_number=samples_count, dimensions=dim)

        # generate 2d classification dataset
        no_objects = self.metadata.no_object
        if self.metadata.sizes is None or len(self.metadata.sizes) == 0:
            self.metadata.sizes = [samples_count] * no_objects
        scale = self.metadata.scale

        sphere_dataset.make_intersected(no_objects=no_objects,
                                        sizes=self.metadata.sizes,
                                        scale=scale, save=False, stdev=stdev)

        self.get_common_data(sphere_dataset.X, sphere_dataset.Y,
                             sphere_dataset.suffix)
        del sphere_dataset

    def load_normal_dist(self):
        can_i_translate = self.metadata.dtype_hdler.is_normal_dist()

        if self.metadata.samples_count is None:
            self.metadata.samples_count = 100
        if self.metadata.dimensions is None:
            self.metadata.dimensions = 3

        samples_count = self.metadata.samples_count
        dim = self.metadata.dimensions

        norm_dist_dataset = NormalDistBasedDatasetsGenerator(
            samples_number=samples_count, dimensions=dim, can_i_translate=can_i_translate)

        # generate 2d classification dataset
        no_objects = self.metadata.no_object
        if self.metadata.sizes is None or len(self.metadata.sizes) == 0:
            self.metadata.sizes = [samples_count] * no_objects

        scale = self.metadata.scale
        mean = self.metadata.mean
        st_dev = self.metadata.stdev
        sizes = self.metadata.sizes

        norm_dist_dataset.make_intersected(no_object=no_objects,
                                           sizes=sizes, mean=mean,
                                           stdev=st_dev, scale=scale, save=False)


        self.get_common_data(norm_dist_dataset.X, norm_dist_dataset.Y,
                             norm_dist_dataset.suffix)
        del norm_dist_dataset

    def load_wine(self):
        wine = datasets.load_wine()
        self.metadata.dimensions = 1000
        self.get_common_data(wine.data, wine.target)

        self.metadata.labels = list(wine.target_names)

    def load_breast_cancer(self):
        breast = datasets.load_breast_cancer()

        self.metadata.dimensions = 1000
        self.get_common_data(breast.data, breast.target)

        self.metadata.labels = list(breast.target_names)

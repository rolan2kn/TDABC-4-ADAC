#!/usr/bin/python
# -*- coding: utf-8 -*-
DTYPE_NUMBER = 10


class DatasetTypeHandler:
    IRIS, SWISSROLL, MOON, CIRCLES, SPHERE, NORMAL_DIST, BREAST_CANCER, WINE, KRNN, NONE = range(DTYPE_NUMBER)

    def __init__(self, dtype = None):
        if dtype is None:
            dtype = DatasetTypeHandler.NONE
        self.type = dtype

    def is_a_skitlearn_type(self):
        return self.is_circles() or \
               self.is_moon() or \
               self.is_iris() or \
               self.is_swissroll() or \
               self.is_wine() or \
               self.is_breast_cancer()

    def is_a_uci_type(self):
        return self.is_breast_cancer() or \
               self.is_wine()

    def is_a_generated_type(self):
        return self.is_sphere() or self.is_normal_dist() or self.is_krnn()
    
    def is_iris(self):
        return self.type == DatasetTypeHandler.IRIS

    def is_krnn(self):
        return self.type == DatasetTypeHandler.KRNN

    def is_swissroll(self):
        return self.type == DatasetTypeHandler.SWISSROLL

    def is_moon(self):
        return self.type == DatasetTypeHandler.MOON

    def is_circles(self):
        return self.type == DatasetTypeHandler.CIRCLES

    def is_sphere(self):
        return self.type == DatasetTypeHandler.SPHERE

    def is_normal_dist(self):
        return self.type == DatasetTypeHandler.NORMAL_DIST

    def is_breast_cancer(self):
        return self.type == DatasetTypeHandler.BREAST_CANCER

    def is_wine(self):
        return self.type == DatasetTypeHandler.WINE

    def is_none(self):
        return self.type == DatasetTypeHandler.NONE

    def to_str(self):
        if self.is_iris():
            return "IRIS"

        elif self.is_swissroll():
            return "SWISSROLL"

        elif self.is_moon():
            return "MOON"

        elif self.is_circles():
            return "CIRCLES"

        elif self.is_sphere():
            return "SPHERE"

        elif self.is_normal_dist():
            return "NORMAL"

        elif self.is_breast_cancer():
            return "CANCER"

        elif self.is_wine():
            return "WINE"


        elif self.is_krnn():
            return "KRNN"


        elif self.is_none():
            return "NONE"

        raise Exception("Unknown dataset type")

    def get_supported_dataset_names(self):
        dtype_names = []
        dtype_old = self.type
        for dtype in range(DTYPE_NUMBER):
            self.type = dtype
            dtype_names.append(self.to_str())
        self.type = dtype_old

        return dtype_names

    def __str__(self):
        string = "DatasetTypeHandler => '{0}'".format(self.to_str())

        return string
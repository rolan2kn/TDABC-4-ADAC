#!/usr/bin/python
# -*- coding: utf-8 -*-
# Created by Rolando Kindelan Nuñez at 20-05-21
# correspondence mail rolan2kn@gmail.com 
# Feature: #Enter feature name here
# Enter feature description here

# Scenario: # Enter scenario name here
# Enter steps here
import numpy as np

from dataset_tools.dataset_dimensionality_reduction import DatasetDimensionalityReduction


class DatasetTransformer:
    NONE, STANDARD, SIMPLE, SQUARED, NLOG, NORM, INV, ELOG, REDUCTION = range(9)
    def __init__(self, **kwargs):
        self.S = kwargs.get("data_points", None)
        if self.S is None:
            dataset_handler = kwargs.get("dataset_handler", None)
            if dataset_handler is None:
                raise Exception("You must provide data_points or a dataset_handler")
            self.S = dataset_handler.unify_dataset()

        self.type = kwargs.get("ttype")
        if self.type is None:
            self.type = kwargs.get("data_transformation_type", DatasetTransformer.NONE)
        self.reduction_method = kwargs.get("reduction_method", DatasetDimensionalityReduction.PCA)
        self.desired_components = kwargs.get("number_of_components", 3)

    def execute(self):
        if self.S is None or len(self.S) < 1:
            raise Exception("ERROR!! Invalid point set")
        transformed_points = []
        abs_min = np.abs(np.min(self.S))
        if abs_min == 0:
            abs_min = 1
        if self.type in (DatasetTransformer.SIMPLE, DatasetTransformer.STANDARD):
            avg = np.average(np.array(self.S))
            std = np.std(np.array(self.S))
            transformed_points = ((np.array(self.S) - avg)/std)

        elif self.type == DatasetTransformer.SQUARED:
            transformed_points = np.sqrt(np.array(self.S)+abs_min)

        elif self.type == DatasetTransformer.NLOG:
            transformed_points = np.log10(np.array(self.S)+abs_min)
        elif self.type == DatasetTransformer.ELOG:
            transformed_points = np.log(np.array(self.S)+abs_min)

        elif self.type == DatasetTransformer.NORM:
            for p in self.S:
                transformed_points.append(np.array(p) / np.linalg.norm(p))
        elif self.type == DatasetTransformer.INV:
            transformed_points = np.nan_to_num(np.reciprocal(np.array(self.S)), posinf=0,
                                               nan=0)  # we detect the max value if we need
        elif self.type == DatasetTransformer.REDUCTION:  # to reduce the dimensionality of datasets
            transformed_points = DatasetDimensionalityReduction.execute(X=self.S, rtype=self.reduction_method,
                                                                        components = self.desired_components)
        else:
            transformed_points = self.S

        return transformed_points


#!/usr/bin/python
# -*- coding: utf-8 -*-
import os
import time

import h5py

from utilities import utils
from utilities.register import Register


class ResultHDF5Storer:
    def __init__(self, expected_results, predicted_results, storage_path=None):
        if storage_path is None:
            storage_path = "{0}/docs/".format(utils.get_module_path())
        self.storage_path = storage_path
        self.timestamp = time.time_ns()
        self.data = None
        self.data_to_hdf5(expected_results, predicted_results)

    def get_expected_results(self):
        if self.data is None:
            self.data = self.load_hdf5()

        return self.data['expected']

    def get_predicted_results(self):
        if self.data is None:
            self.data = self.load_hdf5()

        return self.data['predicted']

    def exists_hdf5_data(self):
        return os.path.isfile(self.get_hdf5_fullname())

    def get_hdf5_fullname(self):
        rpath = "{0}results_data".format(self.storage_path)
        fullname = '{0}/results_{1}.h5'.format(rpath, self.timestamp)

        if not os.path.isdir(rpath):
            os.makedirs(rpath)
        return fullname

    def remove(self):
        if self.data is not None:
            self.data.close()

        if self.exists_hdf5_data():
            os.remove(self.get_hdf5_fullname())

    def data_to_hdf5(self, expected_results, predicted_results):
        if self.exists_hdf5_data():
            return

        filename = self.get_hdf5_fullname()
        with h5py.File(filename, "w") as f:
            f.create_dataset("expected", data = expected_results)
            f.create_dataset("predicted", data = predicted_results)

    def load_hdf5(self):
        if self.data is not None:
            return self.data

        if not self.exists_hdf5_data():
            Register.add_error_message("ERROR: You should create the result hdf5 file")
            raise Exception("ERROR: You should create the result hdf5 file")

        filename = self.get_hdf5_fullname()

        h5 = h5py.File(filename, 'r')

        return h5
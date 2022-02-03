#!/usr/bin/python
# -*- coding: utf-8 -*-
import os
import time
import numpy as np

from dataset_tools.dataset_types import DatasetTypeHandler
from utilities import utils
from utilities.common_generators import CommonDataGenerator
"""
It holds all defined settings concerning the data set. 
"""
class DatasetSettingHandler:
    def __init__(self, **kwargs):
        self.__dict__['args'] = kwargs

    @staticmethod
    def get_real_param_name(related_name):
        params = []

        return params

    def configure_dimension(self):
        dim = self.args.get("desired_dimension", None)
        if dim is None:
            dim = self.args.get("dimensions", None)
        if dim is None:
            dim = CommonDataGenerator.random_int(2, 4)

        self.args.update({"dimensions": dim})

        return dim

    def configure_dataset_type(self):
        data_type = self.args.get("dataset_type", None)
        if data_type is None:
            data_type = self.args.get("desired_data", None)
            if data_type is None:
                dtype = self.args.get("dtype_hdler", None)

                if dtype is None:
                    self.dtype_hdler = DatasetTypeHandler(dtype=DatasetTypeHandler.NONE)

        else:
            self.dtype_hdler = DatasetTypeHandler(dtype=data_type)

        return self.dtype_hdler

    def configure_samples_count(self):
        samples_count = self.args.get("samples_count", None)
        if samples_count is None:
            samples_count = self.args.get("samples_number", None)
        if samples_count is None:
            size = [100, 500, 1000, 5000]
            opt = CommonDataGenerator.random_int(0, 4) % 4
            samples_count = size[opt]
            del size

        self.args.update({"samples_count": samples_count})

        return samples_count

    def configure_no_object(self):
        no_obj = self.args.get("no_object", None)
        if no_obj is None:
            no_obj = self.args.get("no_objects", None)
            if no_obj is None:
                no_obj = CommonDataGenerator.random_int(2, 6)

        self.args.update({"no_object": no_obj})

        return no_obj

    def configure_mean(self):
        mean = self.args.get("mean", None)

        if mean is None:
            mean = CommonDataGenerator.random_float(0, 1)

        self.args.update({"mean": mean})

        return mean

    def configure_st_dev(self):
        st_dev = self.args.get("stdev", None)
        if st_dev is None:
            st_dev = self.args.get("st_dev", None)

        if st_dev is None:
            st_dev = CommonDataGenerator.random_float(0, 1)

        self.args.update({"stdev": st_dev})

        return st_dev

    def configure_sizes(self):
        sizes = self.args.get("sizes", None)
        if sizes is None:
            samples_count = self.configure_samples_count()
            no_obj = self.configure_no_object()

            sizes = []
            for _ in range(no_obj):
                factor = CommonDataGenerator.random_int(1, 4)
                sizes.append(int(samples_count / factor))

            self.args.update({"sizes": sizes})
        return sizes

    def configure_scale(self):
        self.args.update({"scale": self.args.get("scale", 1)})

        return self.args.get("scale")

    def configure_noise(self):
        noise = self.args.get("noise", None)

        if noise is None:
            noise = 0

        self.args.update({"noise": noise})
        return noise

    def configure_suffix(self):
        return self.args.get("suffix", self.configure_dataset_name())

    def configure_time(self):
        time_suffix = self.args.get("time_suffix", None)
        if time_suffix is None:
            time_suffix = time.strftime("%y.%m.%d_%H.%M.%S")

        self.args.update({"time_suffix": time_suffix})

        return time_suffix

    def get_folder(self):
        folder = self.args.get("folder", None)
        if folder is None:
            folder = "docs/CLASSIFIER_EVALUATION"
        self.args.update({"folder": folder})

        return folder

    def configure_path(self):
        new_path = self.args.get("path", None)
        if new_path is None:
            dataset_name = self.configure_dataset_name()
            dataset_time_suffix = self.configure_time()
            new_path = "{2}/{0}/{1}/".format(dataset_name, dataset_time_suffix,
                                             self.get_folder())

        self.args.update({"path": new_path})

        return new_path

    def get_current_name(self):
        path = "{0}/{1}".format(utils.get_module_path(), self.configure_path())
        if not os.path.isdir(path):
            os.makedirs(path)

        if self.suffix is None:
            self.suffix = "{0}_default_args".format(self.dataset_name())

        current_name = "{0}{2}__{1}.png".format(path, self.suffix,
                                                self.configure_time())
        self.args.update({"current_name": current_name})

        return current_name

    def configure_dataset_name(self):
        self.dataset_name = self.dtype_hdler.to_str()

        return self.dataset_name

    def save_properties(self):
        current_name = self.get_current_name()
        pos = current_name.find(".png")
        if pos != -1:
            filename_noext = current_name[:pos]

        filename = "{0}.txt".format(filename_noext)

        dataprop = open(filename, "w")

        self.sizes = self.splitter.real_samples_by_label()
        self.dimensions = self.splitter.real_dimension()

        dataprop.write("NAME: {0}\n".format(self.args.get("dataset_name")))
        dataprop.write(
            "DIMENSIONS: {0}\n".format(self.args.get("dimensions")))
        dataprop.write("SAMPLES_COUNT: {0}\n".format(self.samples_count))
        dataprop.write("TOTAL_SIZE: {0}\n".format(self.splitter.dataset_size()))
        dataprop.write("SIZES: {0}\n".format(self.sizes))
        dataprop.write("IMBALANCED: {0}\n".format(self.imbalanced))
        dataprop.write("IMBALANCED_RATIO: {0}\n".format(self.imbalanced_ratio))
        dataprop.write("NO_LABELS: {0}\n".format(self.no_object))
        dataprop.write("MEAN: {0}\n".format(self.mean))
        dataprop.write("ST_DEV: {0}\n".format(self.stdev))
        dataprop.write("NOISE: {0}\n".format(self.noise))
        dataprop.write("TRANSFORMATION_TYPE: {0}\n".format(self.transformation_type))

        if self.should_export_2_csv:
            csvfilename = "{0}.csv".format(filename_noext)
            self.splitter.export_2_csv(csvfilename)
        dataprop.close()

    def configure_dimensionality_reduction(self):
        reduce = self.args.get("reduction_method", None)

        number_of_components = self.args.get("number_of_components")
        if number_of_components is None :
            number_of_components = np.sqrt(self.dimensions) if reduce is not None else self.dimensions

        self.args.update({"reduction_method": reduce})
        self.args.update({"number_of_components": number_of_components})

    def execute(self):
        self.configure_dataset_type()
        self.configure_dimension()
        self.configure_samples_count()
        self.configure_noise()
        self.configure_no_object()
        self.configure_sizes()
        self.configure_mean()
        self.configure_st_dev()
        self.configure_scale()
        self.configure_dataset_name()
        self.configure_path()
        self.configure_time()

        return self.args

    def cleanup(self):
        del self.args

    def __getattr__(self, item):
        if item in self.__dict__['args']:
            return self.__dict__['args'][item]

    def __setattr__(self, key, value):
        self.__dict__['args'].update({key: value})

    def __del__(self):
        self.cleanup()

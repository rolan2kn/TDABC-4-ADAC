from dataset_tools.dataset_handler import DatasetHandler
from dataset_tools.dataset_transformer import DatasetTransformer
from dataset_tools.dataset_types import DatasetTypeHandler


from dataset_tools.dataset_settings import DatasetSettingHandler
from utilities.register import Register

from dataset_tools.dataset_plotter import DatasetPlotter
from utilities import utils


class BuildDataset4Test:

    def __init__(self, **kwargs):
        data_type = kwargs.get("dataset_type", DatasetTypeHandler())

        folder = kwargs.get("folder", "docs/CLASSIFIER_EVALUATION")
        time_suffix = kwargs.get("time_suffix", None)

        can_i_show = kwargs.get("can_i_show", False)
        plot_all = kwargs.get("plot_all", True)
        plot_2d = kwargs.get("plot_2d", True)
        self.meta_info = {"can_i_show": can_i_show,
                          "plot_all": plot_all,
                          "plot_2d": plot_2d}

        self.dataset_metadata = DatasetSettingHandler()
        self.dataset_metadata.dtype_hdler = data_type
        self.dataset_metadata.should_export_2_csv = kwargs.get("should_export_2_csv", False)
        self.dataset_metadata.data_transformation_type = kwargs.get("data_transformation_type", DatasetTransformer.NONE)
        self.dataset_metadata.reduction_method = kwargs.get("reduction_method", None)
        self.dataset_metadata.number_of_components = kwargs.get("number_of_components", None)
        self.data_transformation_type = kwargs.get("data_transformation_type", DatasetTransformer.NONE)
        self.dataset_metadata.data_index = kwargs.get("data_index", None)
        if time_suffix is not None:
            self.dataset_metadata.time_suffix = time_suffix
        self.dataset_handler = None
        self.dataset_metadata.folder = folder

    def build_dataset(self):
        Register.destroy()
        Register.create("registro.log")
        self.dataset_handler = DatasetHandler(self.dataset_metadata)
        self.dataset_handler.load_dataset()

        DatasetPlotter(self.dataset_handler).draw_data(**self.meta_info)
        self.dataset_metadata.save_properties()

        Register.destroy()
        Register.create("registro.log", self.dataset_metadata.path)

    def build_IRIS(self):
        self.dataset_metadata.dimensions = 4
        self.dataset_metadata.imbalanced = False

    def build_SWISSROLL(self):
        self.dataset_metadata.dimensions = 3
        self.dataset_metadata.no_object = 6
        self.dataset_metadata.noise = 10
        self.dataset_metadata.samples_count = 300
        self.dataset_metadata.imbalanced = False

    def build_KRNN(self):
        data_index = 1 if self.dataset_metadata.data_index is None else self.dataset_metadata.data_index
        self.dataset_metadata.desired_dimension = 2,
        self.dataset_metadata.no_object = 2,
        self.dataset_metadata.mean = [0, 2.0],
        self.dataset_metadata.stdev = [1, 2],
        self.dataset_metadata.sizes = [50, 50*data_index],
        self.dataset_metadata.samples_count = 50,
        self.dataset_metadata.imbalanced = True

    def build_MOON(self):
        self.dataset_metadata.dimensions = 2
        self.dataset_metadata.noise = 10
        self.dataset_metadata.samples_count = 200

    def build_CIRCLES(self):
        self.dataset_metadata.dimensions = 2
        self.dataset_metadata.noise = 3
        self.dataset_metadata.samples_count = 50

    def build_SPHERE(self):
        self.dataset_metadata.dimensions = 3
        self.dataset_metadata.no_object = 5
        self.dataset_metadata.imbalanced = True
        self.dataset_metadata.samples_count = 500
        self.dataset_metadata.mean = 0.3
        self.dataset_metadata.stdev = 0.147
        self.dataset_metadata.sizes = [500, 100, 25, 16, 12],

    def build_NORMAL_DIST(self):
        self.dataset_metadata.desired_dimension = 350,
        self.dataset_metadata.no_object = 5,
        self.dataset_metadata.mean = [0, 0.3, 0.18, 0.67, 0],
        self.dataset_metadata.stdev = 0.3,
        self.dataset_metadata.sizes = [60, 10, 50, 100, 80],
        self.dataset_metadata.samples_count = 50,
        self.dataset_metadata.imbalanced = True

    def build_WINE(self):
        self.dataset_metadata.data_transformation_type = DatasetTransformer.ELOG

    def build_BREAST(self):
        self.dataset_metadata.data_transformation_type = DatasetTransformer.ELOG

    def execute(self):
        try:
            if self.dataset_metadata.dtype_hdler.is_iris():
                self.build_IRIS()
            elif self.dataset_metadata.dtype_hdler.is_swissroll():
                self.build_SWISSROLL()
            elif self.dataset_metadata.dtype_hdler.is_moon():
                self.build_MOON()
            elif self.dataset_metadata.dtype_hdler.is_circles():
                self.build_CIRCLES()
            elif self.dataset_metadata.dtype_hdler.is_normal_dist():
                self.build_NORMAL_DIST()
            elif self.dataset_metadata.dtype_hdler.is_sphere():
                self.build_SPHERE()
            elif self.dataset_metadata.dtype_hdler.is_wine():
                self.build_WINE()
            elif self.dataset_metadata.dtype_hdler.is_breast_cancer():
                self.build_BREAST()
            elif self.dataset_metadata.dtype_hdler.is_krnn():
                self.build_KRNN()

            self.build_dataset()
            return self.dataset_handler
        except BaseException as e:
            Register.add_error_message(e)
            print("ERROR global: {0}".format(e))


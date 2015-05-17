import os
import yaml
import shutil
from collections import namedtuple, OrderedDict
import importlib
from sgd import StochasticGradient

Metric = namedtuple("Metric", ["records", "iterations", "time", "memory", "logloss", "clicks", "not_clicks"])


class MetricsGenerator(object):
    def __init__(self, stream):
        self.stream = stream

    def __call__(self):
        self.stream.readline()
        for line in self.stream:
            records, iteration, time_for_iterations, memory, logloss, clicks, not_clicks = line.split("\t")
            records = int(records)
            iteration = int(iteration)
            time_for_iterations = int(time_for_iterations)
            memory = int(memory) if memory is not "-" else None
            logloss = float(logloss) if logloss is not "-" else None
            clicks = int(clicks)
            not_clicks = int(not_clicks)
            yield Metric(records, iteration, time_for_iterations, memory, logloss, clicks, not_clicks)


def create_dir_if_not_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)


def save_metrics(path, metrics):
    for metric in metrics:
        with open(path, 'a', 0) as f:
            f.write("%s\t%s\t%s\t%s\t%s\t%s\t%s\n" % (
                metric.records,
                metric.iterations,
                metric.time,
                metric.memory if metric.memory is not None else '-',
                metric.logloss if metric.logloss is not None else '-',
                metric.clicks,
                metric.not_clicks)
            )


def init_metrics(path):
    with open(path, 'w', 0) as f:
        f.write("# Records\tIterations\tTime\tMemory\tLogloss\tClicks\tNotClicks\n")


# class BaseLocalFileStorage(object):
#     def __init__(self, path):
#         self.root_path = path
#         self.images_dir_path = os.path.join(self.root_path, "images")
#         self.third_part_resources = os.path.join(path, 'resources')
#
#     def init(self):
#         create_dir_if_not_exists(self.images_dir_path)
#
#     def __image_path(self, identifier):
#         return "%s.png" % os.path.join(self.images_dir_path, identifier)
#
#     def third_part_resource_path(self, identifier):
#         return os.path.join(self.third_part_resources, str(identifier))
#
#     def get_batches_filenames(self, identifier):
#         batches_template = os.path.join(self.third_part_resource_path(identifier), "*.batch")
#         return glob(batches_template)
#
#     def get_dictionary_filename(self, identifier):
#         return os.path.join(self.third_part_resource_path(identifier), 'dictionary')
#
#     def save_image(self, image, identifier):
#         identifier = str(identifier)
#         image.savefig(self.__image_path(identifier))


class LocalFileStorage(object):
    def __init__(self, path, identifier):
        self.root_path = path
        self.work_path_dir = os.path.join(path, identifier)
        self.__metric_path = os.path.join(self.work_path_dir, "metrics.tsv")
        self.__parameters_path = os.path.join(self.work_path_dir, "parameters.yml")
        self.__weights_path = os.path.join(self.work_path_dir, "weights")
        self.is_initialized_metrics = False
        self.init()


    def init(self):
        create_dir_if_not_exists(self.work_path_dir)

    def save_parameters(self, parameters):
        with open(self.__parameters_path, 'w', 0) as f:
            yaml.dump(parameters, f)

    def get_parameters(self):
        with open(self.__parameters_path, 'r') as f:
            return yaml.load(f)

    def save_metrics(self, metrics):
        if not self.is_initialized_metrics:
            self.is_initialized_metrics = True
            init_metrics(self.__metric_path)
        save_metrics(self.__metric_path, metrics)

    def get_metrics(self):
        with open(self.__metric_path, 'r') as f:
            return [metric for metric in MetricsGenerator(f)()]

    def clean(self):
        if os.path.exists(self.work_path_dir):
            shutil.rmtree(self.work_path_dir)

    @staticmethod
    def weight_storage_to_str(weights_storage):
        return "%s.%s" % (weights_storage.module, weights_storage.__class__.__name__)

    @staticmethod
    def str_to_weight_storage(weights_storage_str):
        module_name, _, function_name = weights_storage_str.rpartition(".")
        module = importlib.import_module(module_name)
        return getattr(module, function_name)()

    def dump_weights_storage(self, weights_storage):
        with open(self.__weights_path, 'w') as f:
            weights_storage.save(f)

    def load_weights_storage(self):
        parameters = self.get_parameters()
        weights_storage = self.str_to_weight_storage(parameters["weights_storage"])

        with open(self.__weights_path, 'r') as f:
            weights_storage.load(f)

        self.is_initialized_metrics = os.path.exists(self.__metric_path)

        return weights_storage
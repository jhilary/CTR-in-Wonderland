import os
import yaml
import shutil
import importlib
import cPickle

from collections import namedtuple
from ciw.utils import du

Metric = namedtuple("Metric", ["records", "iterations", "time", "memory", "logloss", "clicks", "not_clicks", "features", "disk_usage"])
Prediction = namedtuple("Prediction", ["record_id", 'value'])


class MetricsGenerator(object):
    def __init__(self, stream):
        self.stream = stream

    def __call__(self):
        self.stream.readline()
        for line in self.stream:
            records, iteration, time_for_iterations, memory, logloss, clicks, not_clicks, features, disk_usage = line.split("\t")
            records = int(records)
            iteration = int(iteration)
            time_for_iterations = int(time_for_iterations)
            memory = int(memory) if memory is not "-" else None
            logloss = float(logloss) if logloss is not "-" else None
            clicks = int(clicks)
            not_clicks = int(not_clicks)
            features = int(features)
            disk_usage = float(disk_usage)
            yield Metric(records, iteration, time_for_iterations, memory, logloss, clicks, not_clicks, features, disk_usage)


def create_dir_if_not_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)


def save_metrics(path, metrics):
    for metric in metrics:
        with open(path, 'a', 0) as f:
            f.write("%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" % (
                metric.records,
                metric.iterations,
                int(metric.time),
                metric.memory if metric.memory is not None else '-',
                metric.logloss if metric.logloss is not None else '-',
                metric.clicks,
                metric.not_clicks,
                metric.features,
                metric.disk_usage)
            )

def save_predictions(path, predictions):
    for prediction in predictions:
        with open(path, 'a', 0) as f:
            f.write("%s,%s\n" % (
                prediction.record_id,
                prediction.value)
            )

def init_predictions(path):
    with open(path, 'w', 0) as f:
        f.write("ID,Prediction\n")

def init_metrics(path):
    with open(path, 'w', 0) as f:
        f.write("# Records\tIterations\tTime\tMemory\tLogloss\tClicks\tNotClicks\tFeatures\tDiskUsage\n")


class LocalFileStorage(object):
    def __init__(self, path, identifier, clean_before_use=False):
        self.root_path = path
        self.work_path_dir = os.path.join(path, identifier)
        self.berkeley_database_path = os.path.join(self.work_path_dir, "bk")
        self.__parameters_path = os.path.join(self.work_path_dir, "parameters.yml")
        self.__weights_path = os.path.join(self.work_path_dir, "weights")
        self._model_path = os.path.join(self.work_path_dir, "model")
        self.is_initialized_metrics = set()
        self.is_initialized_predictions = set()
        self.clean_before_use = clean_before_use
        self.init()

    def __metrics_path(self, identifier='metrics.tsv'):
        return os.path.join(self.work_path_dir, identifier)

    def __predictions_path(self, identifier='predictions.tsv'):
        return os.path.join(self.work_path_dir, identifier)

    def init(self):
        if self.clean_before_use and os.path.exists(self.berkeley_database_path):
            shutil.rmtree(self.berkeley_database_path)
        create_dir_if_not_exists(self.work_path_dir)
        create_dir_if_not_exists(self.berkeley_database_path)

    def save_parameters(self, parameters):
        with open(self.__parameters_path, 'w', 0) as f:
            yaml.dump(parameters, f)

    def get_parameters(self):
        with open(self.__parameters_path, 'r') as f:
            return yaml.load(f)

    def save_metrics(self, metrics, identifier='metrics.tsv'):
        metrics_path = self.__metrics_path(identifier)
        if identifier not in self.is_initialized_metrics:
            self.is_initialized_metrics.add(identifier)
            init_metrics(metrics_path)
        save_metrics(metrics_path, metrics)

    def get_metrics(self, identifier='metrics.tsv'):
        with open(self.__metrics_path(identifier), 'r') as f:
            return [metric for metric in MetricsGenerator(f)()]

    def save_predictions(self, predictions, identifier='predictions.tsv'):
        predictions_path = self.__predictions_path(identifier)
        if identifier not in self.is_initialized_predictions:
            self.is_initialized_predictions.add(identifier)
            init_predictions(predictions_path)
        save_predictions(predictions_path, predictions)


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

    def save(self, model):
        with open(self._model_path, 'w') as f:
            cPickle.dump(model, f)

    def load(self):
        with open(self._model_path, 'r') as f:
            return cPickle.load(f)

    def load_weights_storage(self):
        parameters = self.get_parameters()
        weights_storage = self.str_to_weight_storage(parameters["weights_storage"])

        with open(self.__weights_path, 'r') as f:
            weights_storage.load(f)

        self.is_initialized_metrics = os.path.exists(self.__metrics_path())

        return weights_storage

    def get_model_size(self):
        total_size = 0
        total_size += du(self.berkeley_database_path)
        if os.path.exists(self._model_path):
            total_size += du(self._model_path)
        return total_size
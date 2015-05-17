# -*- coding: utf-8 -*-

import fileinput
import resource
import importlib
import platform
import random
import argparse
import pickle
from functools import partial
from datetime import datetime
from collections import defaultdict
from parser import RecordsGenerator
from sgd import StochasticGradient
from storage import LocalFileStorage, Metric
import sys

class DictStorage(defaultdict):
    module = "main"

    def __init__(self, *args, **kwargs):
        super(DictStorage, self).__init__(dict)

    @staticmethod
    def load(f):
        return pickle.load(f)

    def save(self, f):
        pickle.dump(self, f)


def get_memory_usage():
    if platform.system() == u'Linux':
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss


def filter_not_clicks(record, subsampling_rate):
    factors, label = record
    return label.value == 1 or random.random() < subsampling_rate


def learn(records_generator, storage, model, storage_metrics_dumping_depth):
    start_time = datetime.now()
    for record in records_generator():
        model.learn(record)
        if model.iterations % storage_metrics_dumping_depth == 0:
            metric = Metric(records_generator.counter, model.iterations, (datetime.now() - start_time).seconds, get_memory_usage(), model.progressive_validation_logloss, model.clicks, model.not_clicks) # total_seconds?
            storage.save_metrics([metric])
            storage.dump_weights_storage(SG.weights_storage)
    #save metrics at the end of all iterations
    if model.iterations % storage_metrics_dumping_depth != 0:
        metric = Metric(records_generator.counter_filtered, model.iterations, (datetime.now() - start_time).seconds, get_memory_usage(), model.progressive_validation_logloss, model.clicks, model.not_clicks) # total_seconds?
        storage.save_metrics([metric])
        storage.dump_weights_storage(SG.weights_storage)

    # собрать модель можно будет из весов и параметров
    #storage.load_model()

def str_to_weight_storage(weights_storage_str):
    module_name, _, function_name = weights_storage_str.rpartition(".")
    module = importlib.import_module(module_name)
    return getattr(module, function_name)()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CTR in Wonderland')
    parser.add_argument('--algorithm', required=False, default="sg", type=str, help='algorithms for descent: sg, ftrl_proximal')
    parser.add_argument('--add-bias', required=False, action="store_true", help='Toggle adding bias feature')
    parser.add_argument('--weights-storage', required=False, default='main.DictStorage', help='Weights storage class')
    parser.add_argument('--rate', required=False, default="basic", type=str, help='functions for rate: basic, ada_grad, constant')
    parser.add_argument('--rate-constant-alpha', required=False, default=1, type=float, help='alpha for constant rate, works only if rate is constant')
    parser.add_argument('--rate-ada-grad-alpha', required=False, default=1, type=float, help='alpha for ada_grad, works only if rate is ada_grad')
    parser.add_argument('--rate-ada-grad-beta', required=False, default=1, type=float, help='beta for ada_grad, works only if rate is ada_grad')
    parser.add_argument('--ftrl-proximal-lambda-one', required=False, default=1, type=int, help='lambda_1 for ftrl_proximal, works only if algorithm is ftrl_proximal')
    parser.add_argument('--ftrl-proximal-lambda-two', required=False, default=1, type=int,  help='lambda_2 for ftrl_proximal, works only if algorithm is ftrl_proximal')
    parser.add_argument('--feature-filter', required=False, default="none", type=str, help='Filter features: none, poisson, bloom')
    parser.add_argument('--subsampling', required=False, action="store_true", help='Toggle making subsampling for some group')
    parser.add_argument('--subsampling-label', required=False, default=0, type=int, help='Label for subsampling, works only if subsampling is turned on')
    parser.add_argument('--subsampling-rate', required=False, default=0.35, type=float, help='Rate for subsampling, works only if subsampling is turned on')
    parser.add_argument('--progressive-validation', required=False, action="store_true", help='Toggle progressive validation')
    parser.add_argument('--progressive-validation-depth', required=False, default=100000, type=int, help='Depth of progressive validation, works only if progressive-validation is turned on')
    parser.add_argument('--storage-path', required=False, type=str, default=".", help='Path to working storage')
    parser.add_argument('--storage-label', required=False, type=str, default="v0", help='Label of current model in storage')
    parser.add_argument('--storage-metrics-dumping-depth', required=False, type=int, default=100000, help='Iterations before dump metrics in storage')
    parser.add_argument('--missing-plain-features', required=False, type=str, default="zero", help='Function for replace missing plain features: zero, average')
    parser.add_argument('--normalize-plain-features', required=False, action="store_true", help='Use NG algorithm for plain features normalization')

    args = parser.parse_args()

    storage = LocalFileStorage(args.storage_path, args.storage_label)
    storage.save_parameters(sorted(args.__dict__.iteritems(), key=lambda v: v[0]))

    args.weights_storage = str_to_weight_storage(args.weights_storage)
    if args.subsampling:
        records_filter = partial(filter_not_clicks, subsampling_rate=args.subsampling_rate)
        records_generator = RecordsGenerator(sys.stdin, records_filter)
    else:
        records_generator = RecordsGenerator(sys.stdin)

    SG = StochasticGradient(**args.__dict__)

    learn(records_generator, storage, SG, args.storage_metrics_dumping_depth)
# -*- coding: utf-8 -*-

import fileinput
import resource
import platform
import random
import math
import pickle
from datetime import datetime
from collections import namedtuple, defaultdict
from parser import RecordsGenerator
from sgd import StochasticGradient
from storage import LocalFileStorage, Metric
# def eta(sg):
#     return 1./math.sqrt(sg.iteration_number)


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


def main():
    path = "/home/ilariia/CTR-in-Wonderland"
    storage = LocalFileStorage(path, 'v2')
    weight_storage = DictStorage()
    subsampling_not_clicks_rate = 0.3
    records_generator = RecordsGenerator(fileinput.input())
    iterations_before_dumping_metrics = 10000

    SG = StochasticGradient(weight_storage, scales_not_clicks=1/subsampling_not_clicks_rate, rate="ada_grad")

    start_time = datetime.now()
    for record in records_generator():
        factors, label = record
        if label == 0 and random.random() > subsampling_not_clicks_rate:
            continue
        SG.learn(record)
        if SG.iterations % iterations_before_dumping_metrics == 0:
            metric = Metric(SG.iterations, (datetime.now() - start_time).seconds, get_memory_usage(), SG.progressive_validation_logloss, SG.clicks, SG.not_clicks) # total_seconds?
            storage.save_metrics([metric])

    #save metrics at the end of all iterations
    metric = Metric(SG.iterations, (datetime.now() - start_time).seconds, get_memory_usage(), SG.progressive_validation_logloss, SG.clicks, SG.not_clicks) # total_seconds?
    storage.save_metrics([metric])

    storage.save_model(SG) # в формате sparsehash, че он там читает? Можно слить вообще прямо в бинарном виде, если он позволяет

    # собрать модель можно будет из весов и параметров
    storage.load_model()

if __name__ == "__main__":
    main()
# -*- coding: utf-8 -*-

import fileinput
import resource
import platform
import math
import pickle
from datetime import datetime
from collections import namedtuple
from parser import RecordsGenerator
from sgd import StochasticGradient, LearnMetrics
from storage import LocalFileStorage, Metric
# def eta(sg):
#     return 1./math.sqrt(sg.iteration_number)


class DictStorage(dict):
    module = "main"

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
    storage = LocalFileStorage(path, 'v0')
    weight_storage = DictStorage()
    records_generator = RecordsGenerator(fileinput.input())
    iterations_before_dumping_metrics = 10000


    SG = StochasticGradient(weight_storage)

    iterations_counter = 0
    start_time = datetime.now()
    result = LearnMetrics(None, None)
    for record in records_generator():
        result = SG.predict(record)
        SG.learn(record)
        iterations_counter += 1
        if iterations_counter % iterations_before_dumping_metrics == 0:
            metric = Metric(iterations_counter, (datetime.now() - start_time).seconds, get_memory_usage(), result.logloss) # total_seconds?
            storage.save_metrics([metric])

    #save metrics at the end of all iterations
    metric = Metric(iterations_counter, (datetime.now() - start_time).seconds, get_memory_usage(), result.logloss) # total_seconds?
    storage.save_metrics([metric])

    storage.save_model(SG) # в формате sparsehash, че он там читает? Можно слить вообще прямо в бинарном виде, если он позволяет

    # собрать модель можно будет из весов и параметров
    storage.load_model()

if __name__ == "__main__":
    main()
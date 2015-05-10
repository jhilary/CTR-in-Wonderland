# -*- coding: utf-8 -*-

import fileinput
import resource
import platform
import math
from threading import Thread
from Queue import Queue
import pickle
import concurrent
import multiprocessing.dummy
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


class LearnerConsumer(multiprocessing.dummy.Process):

    def __init__(self, records_queue, start_time):
        super(LearnerConsumer, self).__init__()
        path = "/home/ilariia/CTR-in-Wonderland"
        self.storage = LocalFileStorage(path, 'v0')
        weight_storage = DictStorage()
        self.SG = StochasticGradient(weight_storage)
        self.iterations_before_dumping_metrics = 10000
        self.records_queue = records_queue
        self.start_time = start_time

    def run(self):
        result = LearnMetrics(None, None)
        while True:
            record = self.records_queue.get()
            if record is None:
                self.records_queue.task_done()
                break
            result = self.SG.predict(record)
            self.SG.learn(record)
            if self.SG.iterations % self.iterations_before_dumping_metrics == 0:
                metric = Metric(self.SG.iterations, (datetime.now() - self.start_time).seconds, get_memory_usage(), result.logloss) # total_seconds?
                self.storage.save_metrics([metric])
            self.records_queue.task_done()

        #save metrics at the end of all iterations
        metric = Metric(self.SG.iterations, (datetime.now() - self.start_time).seconds, get_memory_usage(), result.logloss) # total_seconds?
        self.storage.save_metrics([metric])

        self.storage.save_model(self.SG) # в формате sparsehash, че он там читает? Можно слить вообще прямо в бинарном виде, если он позволяет
        self.storage.load_model()

def main():


    start_time = datetime.now()

    records_queue = multiprocessing.dummy.JoinableQueue(1000)

    learner_consumer = LearnerConsumer(records_queue, start_time)
    learner_consumer.start()

    records_generator = RecordsGenerator(fileinput.input())
    for record in records_generator():
        records_queue.put(record)
    records_queue.put(None)


    records_queue.join()



if __name__ == "__main__":
    main()
# -*- coding: utf-8 -*-
import sys
import random
import argparse
import pickle
import os
import cPickle
import marshal
from collections import MutableMapping
from functools import partial
from collections import defaultdict
from ciw.parser import RecordsGenerator
from ciw.sgd import StochasticGradient
from ciw.storage import LocalFileStorage
from ciw.utils import str_to_function
from ciw.operations import predict, learn, validate

#import bsddb

class DictStorage(defaultdict):
    module = "ciw.main"

    def __init__(self, *args, **kwargs):
        super(DictStorage, self).__init__(dict)

    @staticmethod
    def load(f):
        return pickle.load(f)

    def save(self, f):
        pickle.dump(self, f)

    @property
    def features_count(self):
        return sum([len(features) for namespace, features in self.iteritems()] + [0])


class BerkeleyDictWrapper(MutableMapping):
    def __init__(self, path):
        super(BerkeleyDictWrapper, self).__init__()
        self.bsdb_dict = bsddb.hashopen(path,cachesize=2*1024*1024*1024 - 1)

    def __repr__(self):
        return repr(self.bsdb_dict)

    def __str__(self):
        return str(self.bsdb_dict)

    def __getitem__(self, key):
        return marshal.loads(self.bsdb_dict[key])

    def __setitem__(self, key, value):
        self.bsdb_dict[key] = marshal.dumps(value)

    def __contains__(self, key):
        return key in self.bsdb_dict

    def __delitem__(self, key):
        del self.bsdb_dict[key]

    def __iter__(self):
        return iter(self.bsdb_dict)

    def __len__(self):
        return len(self.bsdb_dict)

    def sync(self):
        self.bsdb_dict.sync()


class BerkeleyStorage(MutableMapping):
    module = "ciw.main"

    def __init__(self, database_folder, *args, **kwargs):
        super(BerkeleyStorage, self).__init__()
        self._database_folder = database_folder
        self._storage = {}

    def __setitem__(self, key, value):
        raise NotImplementedError("This storage does not accept adding values by hand")

    def __getitem__(self, key):
        if key in self._storage:
            return self._storage[key]
        else:
            new_file_path = os.path.join(self._database_folder, key)
            self._storage[key] = BerkeleyDictWrapper(new_file_path)
            return self._storage[key]

    def __contains__(self, key):
        return key in self._storage

    def __delitem__(self, key):
        del self._storage[key]

    def __len__(self):
        return len(self._storage)

    def __iter__(self):
        return iter(self._storage)

    def __repr__(self):
        return repr(self._storage)

    def __str__(self):
        return str(self._storage)

    def __setstate__(self, f):
        self.__dict__.update(f)
        self._storage = {}

    def __getstate__(self):
        for namespace_storage in self._storage.values():
            namespace_storage.sync()
        odict = self.__dict__.copy() # copy the dict since we change it
        del odict['_storage']
        return odict

    @staticmethod
    def load(f):
        pass

    def save(self, f):
        pass

    @property
    def features_count(self):
        return sum([len(features) for namespace, features in self._storage.iteritems()] + [0])


def filter_not_clicks(record, subsampling_rate):
    return record.label.value == 1 or random.random() < subsampling_rate


def filter_rate(record, subsampling_rate):
    return random.random() < subsampling_rate


def ciw():
    parser = argparse.ArgumentParser(description='CTR in Wonderland')
    subparsers = parser.add_subparsers(dest="mode", help='Modes of ciw. Awailable: learn, predict, validate')

    parser_learn = subparsers.add_parser('learn', help='learn model')
    parser_learn.add_argument('--algorithm', required=False, default="sg", type=str, help='algorithms for descent: sg, ftrl_proximal')
    parser_learn.add_argument('--add-bias', required=False, action="store_true", help='Toggle adding bias feature')
    parser_learn.add_argument('--weights-storage', required=False, default='ciw.main.DictStorage', help='Weights storage class')
    parser_learn.add_argument('--rate', required=False, default="basic", type=str, help='functions for rate: basic, ada_grad, constant')
    parser_learn.add_argument('--rate-constant-alpha', required=False, default=1, type=float, help='alpha for constant rate, works only if rate is constant')
    parser_learn.add_argument('--rate-ada-grad-alpha', required=False, default=1, type=float, help='alpha for ada_grad, works only if rate is ada_grad')
    parser_learn.add_argument('--rate-ada-grad-beta', required=False, default=1, type=float, help='beta for ada_grad, works only if rate is ada_grad')
    parser_learn.add_argument('--ftrl-proximal-lambda-one', required=False, default=1, type=int, help='lambda_1 for ftrl_proximal, works only if algorithm is ftrl_proximal')
    parser_learn.add_argument('--ftrl-proximal-lambda-two', required=False, default=1, type=int,  help='lambda_2 for ftrl_proximal, works only if algorithm is ftrl_proximal')
    parser_learn.add_argument('--feature-filter', required=False, default="none", type=str, help='Filter features: none, poisson')
    parser_learn.add_argument('--feature-filter-poisson-min-shows', required=False, default=10, type=int, help='Value of minimum shows for categorical features to process')
    parser_learn.add_argument('--subsampling', required=False, default="none", type=str, help='Type of subsampling: none, rate, hitstat')
    parser_learn.add_argument('--subsampling-label', required=False, default=0, type=int, help='Label for subsampling, works only if subsampling is turned on')
    parser_learn.add_argument('--subsampling-rate', required=False, default=0.35, type=float, help='Rate for subsampling, works only if subsampling is turned on')
    parser_learn.add_argument('--progressive-validation', required=False, action="store_true", help='Toggle progressive validation')
    parser_learn.add_argument('--progressive-validation-depth', required=False, default=100000, type=int, help='Depth of progressive validation, works only if progressive-validation is turned on')
    parser_learn.add_argument('--storage-metrics-dumping-depth', required=False, type=int, default=100000, help='Iterations before dump metrics in storage')
    parser_learn.add_argument('--storage-model-dumping-depth', required=False, type=int, default=100000, help='Iterations before dump metrics in storage')
    parser_learn.add_argument('--missing-plain-features', required=False, type=str, default="zero", help='Function for replace missing plain features: zero, average')
    parser_learn.add_argument('--normalize-plain-features', required=False, action="store_true", help='Use NG algorithm for plain features normalization')
    parser_learn.add_argument('--storage-path', required=False, type=str, default=".", help='Path to working storage')
    parser_learn.add_argument('--storage-label', required=False, type=str, default="v0", help='Label of current model in storage')

    parser_validate = subparsers.add_parser('validate', help='predict model')
    parser_validate.add_argument('--storage-predictions-dumping-depth', required=False, type=int, default=100000, help='Iterations before dump validation metrics')
    parser_validate.add_argument('--storage-predictions-id', required=False, type=str, default='predictions', help='Identifier in storage to dump validation metrics')
    parser_validate.add_argument('--storage-path', required=False, type=str, default=".", help='Path to working storage')
    parser_validate.add_argument('--storage-label', required=False, type=str, default="v0", help='Label of current model in storage')

    parser_predict = subparsers.add_parser('predict', help='predict model')
    parser_predict.add_argument('--storage-predictions-id', required=False, type=str, default='predictions', help='Identifier in storage to dump predictions')
    parser_predict.add_argument('--storage-path', required=False, type=str, default=".", help='Path to working storage')
    parser_predict.add_argument('--storage-label', required=False, type=str, default="v0", help='Label of current model in storage')

    args = parser.parse_args()

    if args.mode == 'learn':
        storage = LocalFileStorage(args.storage_path, args.storage_label, clean_before_use=True)
        storage.save_parameters(sorted(args.__dict__.iteritems(), key=lambda v: v[0]))

        args.weights_storage = str_to_function(args.weights_storage)(database_folder=storage.berkeley_database_path)
        if args.subsampling == 'hitstat':
            records_filter = partial(filter_not_clicks, subsampling_rate=args.subsampling_rate)
            records_generator = RecordsGenerator(sys.stdin, records_filter)
        elif args.subsampling == 'rate':
            records_filter = partial(filter_rate, subsampling_rate=args.subsampling_rate)
            records_generator = RecordsGenerator(sys.stdin, records_filter)
        elif args.subsampling == 'none':
            records_generator = RecordsGenerator(sys.stdin)
        else:
            raise ValueError("Bad subsampling option '%s'. Available: none, rate, hitstat")
        SG = StochasticGradient(**args.__dict__)

        learn(records_generator, storage, SG, args.storage_metrics_dumping_depth, args.storage_model_dumping_depth)

    if args.mode == 'validate':
        storage = LocalFileStorage(args.storage_path, args.storage_label)
        SG = storage.load()
        records_generator = RecordsGenerator(sys.stdin)
        validate(records_generator, SG, args.storage_predictions_dumping_depth, storage, args.storage_predictions_id)

    if args.mode == 'predict':
        storage = LocalFileStorage(args.storage_path, args.storage_label)
        SG = storage.load()
        records_generator = RecordsGenerator(sys.stdin)
        predict(records_generator, SG, storage, args.storage_predictions_id)

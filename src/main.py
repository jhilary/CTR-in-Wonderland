# -*- coding: utf-8 -*-
import sys
import random
import argparse
import pickle
from functools import partial
from collections import defaultdict
from ciw.parser import RecordsGenerator
from ciw.sgd import StochasticGradient
from ciw.storage import LocalFileStorage
from ciw.utils import str_to_function
from ciw.operations import predict, learn, validate


class DictStorage(defaultdict):
    module = "ciw.main"

    def __init__(self, *args, **kwargs):
        super(DictStorage, self).__init__(dict)

    @staticmethod
    def load(f):
        return pickle.load(f)

    def save(self, f):
        pickle.dump(self, f)


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
        storage = LocalFileStorage(args.storage_path, args.storage_label)
        storage.save_parameters(sorted(args.__dict__.iteritems(), key=lambda v: v[0]))

        args.weights_storage = str_to_function(args.weights_storage)()
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

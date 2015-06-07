import math
import random
from collections import deque, defaultdict
from ciw.metrics import ll
from ciw.feature_types import PlainFeature, CategoricalFeature, FeatureInfo



class StochasticGradient(object):

    def basic_rate(self, *args):
        return 1.0/math.sqrt(self.iterations)

    def constant_rate(self, *args):
        return self.rate_constant_alpha

    def ada_grad_rate(self, g_square=0):
        return float(self.rate_ada_grad_alpha)/(self.rate_ada_grad_beta + math.sqrt(g_square))

    def poisson_feature_filter(self, feature):
        if isinstance(feature, CategoricalFeature) and not feature.info.is_used_before:
            return random.random() < 1.0/self.feature_filter_poisson_min_shows
        return True

    def __init__(self, weights_storage, algorithm="sg", add_bias=False, rate="basic", rate_constant_alpha=1, rate_ada_grad_alpha=1,
                 rate_ada_grad_beta=1, ftrl_proximal_lambda_one=0, ftrl_proximal_lambda_two=0, feature_filter="none",
                 feature_filter_poisson_min_shows=10, subsampling="none",
                 subsampling_label=0, subsampling_rate=0.35, progressive_validation=False, progressive_validation_depth=100000, missing_plain_features="zero",
                 normalize_plain_features=False, **kwargs):
        self.weights_storage = weights_storage
        self.algorithm = algorithm
        self.feature_filter = feature_filter
        self.feature_filter_poisson_min_shows = feature_filter_poisson_min_shows
        self.rate = rate
        self.rate_constant_alpha = rate_constant_alpha
        self.rate_ada_grad_alpha = rate_ada_grad_alpha
        self.rate_ada_grad_beta = rate_ada_grad_beta
        self.add_bias = add_bias
        self.ftrl_proximal_lambda_one = ftrl_proximal_lambda_one
        self.ftrl_proximal_lambda_two = ftrl_proximal_lambda_two
        self.subsampling = subsampling
        self.subsampling_label = subsampling_label
        self.subsampling_rate = subsampling_rate
        self.progressive_validation = progressive_validation
        self.progressive_validation_depth = progressive_validation_depth
        self.missing_plain_values = missing_plain_features
        self.normalize_plain_features = normalize_plain_features
        self.iterations = 0
        self.learn = self._choose_algo(self.algorithm)
        self.rate_func = self._choose_rate(self.rate)
        self.feature_filter_func = self._choose_feature_filter(self.feature_filter)

        if self.progressive_validation:
            self.progressive_validation_queue = deque(maxlen=progressive_validation_depth)
        else:
            self.progressive_validation_queue = None

        self.process_missing_values = self._choose_missing_values(self.missing_plain_values)

        if self.missing_plain_values == "average":
            self.average_dict = defaultdict(dict)
        else:
            self.average_dict = None

        if self.normalize_plain_features:
            self.normalizing_s = defaultdict(dict)
            self.normalizing_N = 0
        else:
            self.normalizing_s = None
            self.normalizing_N = None

        self.clicks = 0
        self.not_clicks = 0

        self.feature_counter = 0

    def load_features_info(self, record_factors):
        for namespace, feature in record_factors.iteritems():
            storage_value = self.weights_storage[namespace].get(feature.name)
            if storage_value is None:
                feature.info = FeatureInfo(0, 0, is_used_before=False)
            else:
                feature.info = FeatureInfo(*storage_value, is_used_before=True)

    def save_features_info(self, record_factors):
        for namespace, feature in record_factors.iteritems():
            self.weights_storage[namespace][feature.name] = (feature.info.weight, feature.info.g_square)

    def update_ng_normalize_parameters_and_weights(self, record_factors):
        for namespace, feature in record_factors.iteritems():
            if isinstance(feature, PlainFeature):
                value = self.process_missing_values(namespace, feature)
                s = self.normalizing_s[namespace].get(feature.name, 0)
                if abs(value) > s:
                    updated_weight = float(feature.info.weight * (s ** 2)) / (value ** 2)
                    feature.info.weight = updated_weight
                    self.normalizing_s[namespace][feature.name] = abs(value)
                if abs(value) > 0:
                    self.normalizing_N += float(abs(value) ** 2) / ((self.normalizing_s[namespace][feature.name]) ** 2)

    @property
    def progressive_validation_logloss(self):
        if self.progressive_validation:
            if len(self.progressive_validation_queue) < self.progressive_validation_depth:
                print "WARNING: progressive validation is calculated for %s items" % len(self.progressive_validation_queue)
            if len(self.progressive_validation_queue) == 0:
                print "WARNING: model has not been learned yet so progressive validation logloss returns None"
                return None
            return sum([element[0] for element in self.progressive_validation_queue])/\
                   sum([element[1] for element in self.progressive_validation_queue])
        else:
            return None

    def _learn_ftrl_proximal(self, record):
        self.iterations += 1
        pass

    def _learn_stochastic_gradient(self, record):

        self.iterations += 1

        if record.label.value == 0:
            self.not_clicks += 1
        else:
            self.clicks += 1

        if self.add_bias:
            record.factors["BIAS"] = PlainFeature(1)

        self.load_features_info(record.factors)

        record_weight = 1
        if self.subsampling == "hitstat" and record.label.value == self.subsampling_label:
            record_weight = 1.0/self.subsampling_rate

        if self.normalize_plain_features:
            self.update_ng_normalize_parameters_and_weights(record.factors)

        predicted_label = self.predict_proba(record.factors)

        for namespace, feature in record.factors.iteritems():
            if not self.feature_filter_func(feature):
                continue

            value = self.process_missing_values(namespace, feature)
            g = (record.label.value - predicted_label) * value

            if self.normalize_plain_features and self.normalizing_s[namespace].get(feature.name) is not None:
                features_normalizer = math.sqrt(float(self.iterations)/self.normalizing_N) / self.normalizing_s[namespace][feature.name] ** 2
            else:
                features_normalizer = 1

            feature.info.weight += self.rate_func(feature.info.g_square) * features_normalizer * record_weight * g
            feature.info.g_square += record_weight * (g ** 2)

        if self.progressive_validation:
            self.progressive_validation_queue.append((record_weight * ll([record.label.value],[predicted_label]),record_weight))

        self.save_features_info(record.factors)

    def _choose_algo(self, label):
        if label == "sg":
            return self._learn_stochastic_gradient
        if label == "ftrl_proximal":
            return self._learn_ftrl_proximal
        raise ValueError("There are no algorithm with label %s; Choose: sg, ftrl_proximal" % label)

    def _choose_rate(self, label):
        if label == "basic":
            return self.basic_rate
        if label == "ada_grad":
            return self.ada_grad_rate
        if label == "constant":
            return self.constant_rate
        raise ValueError("There are no rates with label %s; Choose: basic, ada_grad, constant" % label)

    def _choose_feature_filter(self, label):
        if label == "none":
            return lambda feature: True
        if label == "poisson":
            return self.poisson_feature_filter
        raise ValueError("There are no filters with label %s; Choose: none, poisson")

    def zero_missing_value(self, namespace, feature):
        if isinstance(feature, PlainFeature):
            if feature.value is None:
                return 0
        else:
            if feature.value is None:
                raise ValueError("Missing value in namespace %s for feature name %s" % (namespace, feature.name))
        return feature.value

    def average_missing_value(self, namespace, feature):
        value = feature.value
        if isinstance(feature, PlainFeature):
            if value is None:
                value = self.average_dict[namespace].get(feature.name, 0)
            current_average = self.average_dict[namespace].get(feature.name, 0)
            new_average = current_average + 1.0/self.iterations * (value - current_average)
            self.average_dict[namespace][feature.name] = new_average
        else:
            if value is None:
                raise ValueError("Missing value in namespace %s for feature name %s" % (namespace, feature.name))
        return value

    def _choose_missing_values(self, label):
        if label == "zero":
            return self.zero_missing_value
        if label == "average":
            return self.average_missing_value
        raise ValueError("There are no rates with label %s; Choose: zero, average" % label)

    def predict_proba(self, factors):
        total = 0
        for namespace, feature in factors.iteritems():
            value = self.process_missing_values(namespace, feature)
            total += feature.info.weight * value
        try:
            p = 1.0/(1 + math.exp(-1 * total))
            return p / (p + (1-p)/self.subsampling_rate)
        except OverflowError:
            if total > 0:
                return 1
            else:
                return 0

    def predict(self, factors):
        proba = self.predict_proba(factors)
        if proba > 0.5:
            return 1
        return 0

    def __getstate__(self):
        odict = self.__dict__.copy() # copy the dict since we change it
        del odict['process_missing_values']
        del odict['learn']
        del odict['rate_func']
        del odict['feature_filter_func']
        return odict

    def __setstate__(self, dict):
        self.__dict__.update(dict)
        self.process_missing_values = self._choose_missing_values(self.missing_plain_values)
        self.learn = self._choose_algo(self.algorithm)
        self.rate_func = self._choose_rate(self.rate)
        self.feature_filter_func = self._choose_feature_filter(self.feature_filter)

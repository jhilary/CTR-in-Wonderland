import math
from collections import deque, defaultdict
from metrics import ll
from feature_types import PlainFeature, CategoricalFeature, Label


class StochasticGradient(object):
    learn = None
    rate_func = None
    process_missing_values = None

    def basic_rate(self, *args):
        return 1.0/math.sqrt(self.iterations)

    def constant_rate(self, *args):
        return self.rate_constant_alpha

    def ada_grad_rate(self, g_square=0):
        return float(self.rate_ada_grad_alpha)/(self.rate_ada_grad_beta + math.sqrt(g_square))

    def __init__(self, weights_storage, algorithm="sg", add_bias=False, rate="basic", rate_constant_alpha=1, rate_ada_grad_alpha=1,
                 rate_ada_grad_beta=1, ftrl_proximal_lambda_one=0, ftrl_proximal_lambda_two=0, feature_filter="none", subsampling=False,
                 subsampling_label=0, subsampling_rate=0.35, progressive_validation=False, progressive_validation_depth=100000, missing_plain_features="zero",
                 normalize_plain_features=False, **kwargs):
        self.weights_storage = weights_storage
        self.algorithm = algorithm
        self.feature_filter = feature_filter
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

        self.learn = self._choose_algo(algorithm)
        self.rate_func = self._choose_rate(rate)

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

    def update_ng_normalize_parameters_and_weights(self, record_factors):
        for namespace, feature in record_factors.iteritems():
            if isinstance(feature, PlainFeature):
                value = self.process_missing_values(namespace, feature)
                weight, g_square = self.weights_storage[namespace].get(feature.name, (0, 0))
                s = self.normalizing_s[namespace].get(feature.name, 0)
                if abs(value) > s:
                    updated_weight = float(weight * (s ** 2)) / (value ** 2)
                    self.weights_storage[namespace][feature.name] = (updated_weight, g_square)
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

        record_factors, record_label = record

        if record_label.value == 0:
            self.not_clicks += 1
        else:
            self.clicks += 1

        if self.add_bias:
            record_factors["BIAS"] = PlainFeature(1)

        predicted_label = self.predict_proba(record_factors)

        record_weight = 1
        if self.subsampling and record_label.value == self.subsampling_label:
            record_weight = 1.0/self.subsampling_rate

        if self.normalize_plain_features:
            self.update_ng_normalize_parameters_and_weights(record_factors)

        for namespace, feature in record_factors.iteritems():
            value = self.process_missing_values(namespace, feature)
            #print namespace, "value:", value, "Label:", record_label.value, "Predicted label:", predicted_label
            g = (record_label.value - predicted_label) * value * record_weight
            #print "G:", g

            weight, g_square = self.weights_storage[namespace].get(feature.name, (0, 0))

            if self.normalize_plain_features and self.normalizing_s[namespace].get(feature.name) is not None:
                features_normalizer = math.sqrt(float(self.iterations)/self.normalizing_N) / self.normalizing_s[namespace][feature.name] ** 2
            else:
                features_normalizer = 1

            self.weights_storage[namespace][feature.name] = (weight + self.rate_func(g_square) * features_normalizer * g, g_square + (g ** 2))


            #print "Weight:", weight, "New weight", self.weights_storage[namespace][feature.name]
        if self.progressive_validation:
            self.progressive_validation_queue.append((record_weight * ll([record_label.value],[predicted_label]),record_weight))

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
            #print "Feature weight:", self.weights_storage[namespace].get(feature.name, (0,0))[0]
            total += self.weights_storage[namespace].get(feature.name, (0,0))[0] * value
        try:
            return 1.0/(1 + math.exp(-1 * total))
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
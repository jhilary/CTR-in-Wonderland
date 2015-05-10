import math
from collections import deque
from metrics import ll


class StochasticGradient(object):
    learn = None
    rate_func = None
    process_missing_values = None

    def basic_rate(self, *args):
        return 25 * 1.0/math.sqrt(self.iterations)

    def ada_grad_rate(self, g_square=0):
        return 1.0/(self.ada_grad_beta + math.sqrt(g_square))


    def __init__(self, weights_storage, algorithm="sg", feature_filter="none", rate="basic", ada_grad_alpha="basic",
                 ada_grad_beta=1, ftrl_proximal_lambda_1=0, ftrl_proximal_lambda_2=0, subsampling="none",
                 progressive_validation=True, progressive_validation_depth=10000, missing_plain_values="zero", bias=True, scales_not_clicks=1):
        self.weights_storage = weights_storage
        self.algorithm = algorithm
        self.feature_filter = feature_filter
        self.rate = rate
        self.ada_grad_alpha = ada_grad_alpha
        self.ada_grad_beta = ada_grad_beta
        #TODO to params
        self.bias=True
        #TODO make literal one and two
        self.ftrl_proximal_lambda_1 = ftrl_proximal_lambda_1
        self.ftrl_proximal_lambda_2 = ftrl_proximal_lambda_2
        self.subsampling = subsampling
        self.progressive_validation = progressive_validation
        self.progressive_validation_depth = progressive_validation_depth
        #TODO save iterations to parameters
        self.iterations = 0

        self.learn = self._choose_algo(algorithm)
        self.rate_func = self._choose_rate(rate)
        self.progressive_validation_queue = deque(maxlen=progressive_validation_depth)
        self.process_missing_values = self._choose_missing_values(missing_plain_values)
        self.clicks = 0
        self.not_clicks = 0
        self.scales_not_clicks = scales_not_clicks

    @property
    def progressive_validation_logloss(self):
        if len(self.progressive_validation_queue) < self.progressive_validation_depth:
            print "WARNING: progressive validation is calculated for %s items" % len(self.progressive_validation_queue)
        if len(self.progressive_validation_queue) == 0:
            print "WARNING: model has not been learned yet so progressive validation logloss returns None"
            return None
        return sum(self.progressive_validation_queue)/len(self.progressive_validation_queue)

    def _learn_ftrl_proximal(self, record):
        self.iterations += 1
        #print "I'm ftrl proximal"
        pass

    def _learn_stochastic_gradient(self, record):
        self.iterations += 1
        factors, label = record
        if label == 0:
            self.not_clicks += 1
        elif label == 1:
            self.clicks += 1
        else:
            raise ValueError("Bad label %s provided" % label)
        predicted_label = self.predict_proba(factors)
       # print factors
        #print label
        #print self.weights_storage
        factors["BIAS"] = {"0": 1}
        print label, predicted_label
        for namespace, features in factors.iteritems():
            for feature, value in features.iteritems():
                value = self.process_missing_values(value)
                g = (label - predicted_label) * value
                if label == 0:
                    g *= self.scales_not_clicks
                weight, g_square = self.weights_storage[namespace].get(feature, (0, 0))
                self.weights_storage[namespace][feature] = (weight + self.rate_func(g_square) * g, g_square + g ** 2)

        # print self.rate_func()
        # print label, predicted_label
        # print self.weights_storage
                # d.setdefault(word, []).append(x)

        self.progressive_validation_queue.append(ll([label],[predicted_label]))
        #print self.progressive_validation_queue

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
        raise ValueError("There are no rates with label %s; Choose: basic, ada_grad" % label)

    def zero_missing_value(self, value):
        if value is None:
            return 0
        return value

    def average_missing_value(self, value):
        raise NotImplementedError("AAAAA")

    def _choose_missing_values(self, label):
        if label == "zero":
            return self.zero_missing_value
        if label == "average":
            return self.average_missing_value
        raise ValueError("There are no rates with label %s; Choose: zero, average" % label)


    def predict_proba(self, factors):
        total = 0
        for namespace, features in factors.iteritems():
            for feature, value in features.iteritems():
                value = self.process_missing_values(value)
                total += self.weights_storage[namespace].get(feature, (0,0))[0] * value
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
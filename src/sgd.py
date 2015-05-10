from collections import namedtuple
import time
LearnMetrics = namedtuple("LearnMetrics", ["logloss", "auc"])


class StochasticGradient(object):
    learn = None

    def __init__(self, weights_storage, algorithm="sg", feature_filter="none", rate="basic", ada_grad_alpha="basic",
                 ada_grad_beta=1, ftrl_proximal_lambda_1=0, ftrl_proximal_lambda_2=0, subsampling="none",
                 progressive_validation=True, progressive_validation_depth=100):
        self.weights_storage = weights_storage
        self.algorithm = algorithm
        self.feature_filter = feature_filter
        self.rate = rate
        self.ada_grad_alpha = ada_grad_alpha
        self.ada_grad_beta = ada_grad_beta
        #TODO make literal one and two
        self.ftrl_proximal_lambda_1 = ftrl_proximal_lambda_1
        self.ftrl_proximal_lambda_2 = ftrl_proximal_lambda_2
        self.subsampling = subsampling
        self.progressive_validation = progressive_validation
        self.progressive_validation_depth = progressive_validation_depth
        self.iterations = 0
        self.result = LearnMetrics(None, None)

        self.learn = self._choose_algo(algorithm)

    def _learn_ftrl_proximal(self, record):
        self.iterations += 1
        #print "I'm ftrl proximal"
        pass

    def _learn_stochastic_gradient(self, record):
        self.iterations += 1
        time.sleep(0.001)
        #print "I'm stochastic gradient"
        pass

    def _choose_algo(self, label):
        if label == "sg":
            return self._learn_stochastic_gradient
        if label == "ftrl_proximal":
            return self._learn_ftrl_proximal
        raise ValueError("There are no algorithm with label %s; Choose: sg, ftrl_proximal" % label)


    def predict(self, record):
        self.result = LearnMetrics(1, 2)
        return self.result
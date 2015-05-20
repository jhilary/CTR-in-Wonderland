import pandas as pd
import sklearn.linear_model
import sklearn.dummy

from sklearn.metrics import log_loss


def sgd_baseline(X, y):
    c = sklearn.linear_model.SGDClassifier(loss='log', fit_intercept=True, shuffle=False, penalty='none', alpha=1, n_iter=1, eta0=1, learning_rate='invscaling')
    classes = [0, 1]

    predictions = []
    for i in xrange(len(y)):
        if i != 0:
            predicted_label = c.predict_proba(X.irow(i))[0]
            predictions.append(predicted_label)
            #print "Predicted label:", predicted_label
        c.partial_fit(X.irow(i), (y[i],), classes=classes)
        #print "Value", X.irow(i).values, "Label:",y[i]

    return log_loss(y[1:], predictions)


def sgd_not_online_baseline(X,y, X_test, y_test):
    c = sklearn.linear_model.SGDClassifier(loss='log', fit_intercept=True, shuffle=False, penalty='none', alpha=1, n_iter=1, eta0=1, learning_rate='invscaling')
    c.fit(X, y)
    predicted_labels = c.predict_proba(X_test)
    return log_loss(y_test, predicted_labels)


def uniform_baseline(X,y, X_test, y_test):
    dc = sklearn.dummy.DummyClassifier('uniform')
    dc.fit(X, y)

    dummy_predictions = dc.predict_proba(X_test)
    return log_loss(y_test, dummy_predictions)


def most_frequent_baseline(X, y, X_test, y_test):
    dc = sklearn.dummy.DummyClassifier('most_frequent')
    dc.fit(X, y)

    dummy_predictions = dc.predict_proba(X_test)
    return log_loss(y_test, dummy_predictions)


def constant_baseline(y, constant):
    dummy_predictions = [constant] * len(y)
    return log_loss(y, dummy_predictions)


def get_X_y_cat1(filename):

    with open(filename) as f:
        labels = []
        features = []
        f.readline() #headers
        for i in f.readlines():
            l = i.strip().split(",")
            labels.append(int(l[0]))
            features.append(l[1])
    dummies = pd.get_dummies(features)

    return dummies, labels


def get_X_y_num1(filename):

    with open(filename) as f:
        labels = []
        features = []
        f.readline() #headers
        for i in f.readlines():
            l = i.strip().split(",")
            labels.append(int(l[0]))
            features.append(l[38])

    return pd.DataFrame(features), labels

def get_X_y_num1_num2(filename):

    with open(filename) as f:
        labels = []
        features = []
        f.readline() #headers
        for i in f.readlines():
            l = i.strip().split(",")
            labels.append(int(l[0]))
            features.append([l[38], l[39]])

    return pd.DataFrame(features), labels

def get_X_y_num1_b(filename):

    with open(filename) as f:
        labels = []
        features = []
        f.readline() #headers
        for i in f.readlines():
            l = i.strip().split(",")
            labels.append(int(l[0]))
            features.append(1)

    return pd.DataFrame(features), labels

def get_X_y_allnum(filename):

    with open(filename) as f:
        labels = []
        features = []
        f.readline() #headers
        for i in f.readlines():
            l = i.strip().split(",")
            labels.append(int(l[0]))
            features.append([float(i) if i != '' else 0 for i in l[38:] ])

    return pd.DataFrame(features), labels

if __name__ == "__main__":

    X, y = get_X_y_cat1("../resources/learn_100000.txt")

    # Only Category 1 baseline

    print "CAT01 SGD log_loss: %s" % sgd_baseline(X, y)

    # Uniform baseline

    print "Uniform dummy classifier logloss: %s" % uniform_baseline(X[:50000], y[:50000], X[50000:], y[50000:])

    # Most frequent label baseline

    print "Most frequent dummy classifier logloss: %s" % most_frequent_baseline(X[:50000], y[:50000], X[50000:], y[50000:])

    # Constant baseline

    constant = float(sum(y))/len(y)
    print "Constant classifier (with constant %s) logloss: %s" % (constant, constant_baseline(y, constant))

    #Only numerical 1 baseline

    X, y = get_X_y_num1("../resources/learn_100000.txt")
    #X, y = get_X_y_num1_b("../resources/learn_100000.txt")
    print "NUM01 SGD log_loss: %s" % sgd_baseline(X, y)

    X, y = get_X_y_num1_num2("../resources/learn_100000.txt")
    #X, y = get_X_y_num1_b("../resources/learn_100000.txt")
    print "NUM01 NUM02 SGD log_loss: %s" % sgd_baseline(X, y)

    X, y = get_X_y_allnum("../resources/learn_100000.txt")
    print "NUM all SGD log_loss: %s" % sgd_baseline(X, y)

    X, y = get_X_y_allnum("../resources/learn_100000.txt")
    print "NUM all SGD log_loss: %s" % sgd_not_online_baseline(X[:50000], y[:50000], X[50000:], y[50000:])
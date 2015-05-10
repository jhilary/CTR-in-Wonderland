import pandas as pd
import sklearn.linear_model
import sklearn.dummy

from sklearn.metrics import log_loss

if __name__ == "__main__":

    with open("../resources/learn_100000.txt") as f:
        labels = []
        features = []
        f.readline() #headers
        for i in f.readlines():
            l = i.strip().split(",")
            labels.append(l[0])
            features.append(l[1])

    dummies = pd.get_dummies(features)

    # Only Category 1

    c = sklearn.linear_model.SGDClassifier(loss='log', penalty='none', alpha=0, n_iter=1, eta0=1, learning_rate='invscaling')
    c.fit(dummies, labels)
    classes = ['0','1']

    predictions = []
    for i in xrange(len(labels)):
        if i != 0:
            predictions.append(c.predict_proba(dummies.irow(i))[0])
        c.partial_fit(dummies.irow(i), (labels[i],), classes=classes)

    print "CAT01 SGD log_loss: %s" %  log_loss(labels[1:], predictions)

    # Uniform

    dc = sklearn.dummy.DummyClassifier('uniform')
    dc.fit(dummies[:50000], labels[:50000])

    dummy_predictions = dc.predict_proba(dummies[50000:])
    print "Uniform dummy classifier logloss: %s" % log_loss(labels[50000:], dummy_predictions)

    # Most frequent label

    dc = sklearn.dummy.DummyClassifier('most_frequent')
    dc.fit(dummies[:50000], labels[:50000])

    dummy_predictions = dc.predict_proba(dummies[50000:])
    print "Most frequent dummy classifier logloss: %s" % log_loss(labels[50000:], dummy_predictions)
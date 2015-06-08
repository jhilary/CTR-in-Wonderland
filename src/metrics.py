from math import log

def ll(acts, preds):
    epsilon = 1e-15
    ll = 0
    for act, pred in zip(acts, preds):
        if pred < epsilon:
            pred = epsilon
        if pred > 1 - epsilon:
            pred = 1 - epsilon
        ll += act*log(pred) + (1-act)*log(1-pred)
    ll = ll * -1.0 / len(acts)
    return ll

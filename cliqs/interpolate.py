from math import log, exp
import random

from rfutils import err

from .rfunc import safelog, logsumexp, INF

VERBOSE = 2

def baum_welch_iter(weights, scores, data):
    # :: Iterable Float x Dict x Iterable a -> [Float]
    # scores :: { Int (datapoint id) -> Int (model id) -> Float (score) }
    c_ml = [propto(weight * scores[d][m] for m, weight in enumerate(weights))
            for d, _ in enumerate(data)]
    return propto(weight * sum(d[m] for d in c_ml)
                  for m, weight in enumerate(weights))

def propto(xs):
    # :: Iterable a -> [a]
    xs = list(xs)
    norm = sum(xs)
    return [x / norm for x in xs]

def init_weights_uniform(models, data):
    # :: Iterable a -> b -> [Float]
    return propto(1 for _ in models)

def init_weights_random(models, data):
    # :: Iterable a -> b -> Random [Float]
    return propto(random.random() for _ in models)

def baum_welch_iterations(score_fs, data, init_weights):
    # :: Iterable (a -> Float) x Iterable a -> Iterator (Float, [Float])
    score_fs = list(score_fs)
    data = list(data)
    weights = init_weights(score_fs, data)
    scores = [[exp(get_score(*datapoint)) for get_score in score_fs]
              for datapoint in data]
    while True:
        weights = baum_welch_iter(weights, scores, data)
        objective = sum(logsumexp(safelog(weights[m]) + safelog(scores[d][m])
                                  for m, _ in enumerate(weights))
                        for d, _ in enumerate(data))
        yield objective, weights

def monotonic_prefix(xs):
    xs_it = iter(xs)
    old_x = next(xs_it)
    yield old_x
    for x in xs_it:
        if x > old_x:
            yield x
            old_x = x
        else:
            break

def baum_welch_weights(score_fs,
                       data,
                       init_weights=init_weights_uniform):
    # :: Iterable (a -> Float) x Iterable a -> [Float]
    old_lik = -INF
    old_weights = None
    iterations = baum_welch_iterations(score_fs, data, init_weights)
    for lik, weights in iterations:
        if VERBOSE:
            err("Did a Baum-Welch iteration; likelihood: %s" % lik)
        if lik > old_lik:
            old_lik = lik
            old_weights = weights
        else:
            return old_weights

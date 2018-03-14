import random
import bisect
import itertools as it
from math import log, exp, log

from rfutils.compat import *

INF = float('inf')

def logaddexp(x, y):
    return safelog(exp(x) + exp(y))

def logsumexp(xs):
    return safelog(sum(map(exp, xs)))

def safelog(x):
    if x:
        return log(x)
    else:
        return -INF

def weighted_choice_king(weights):
    total = 0
    winner = 0
    for i, w in enumerate(weights):
        total += w
        if random.random() * total < w:
            winner = i
    return winner

def streaming_choice(choices):
    winner = None
    total = 0
    for choice in choices:
        total += 1
        if random.random() * total < 1:
            winner = choice
    return winner

try:
    
    it.accumulate # check if we have python 3 itertools
    
    def sample_weighted_choice(choices_and_weights):
        """ Choose an element of choices (list of object, prob tuples). """        
        if isinstance(choices_and_weights, dict):
            choices_and_weights = choices_and_weights.items()
        choices, weights = zip(*choices_and_weights)
        assert choices
        cumdist = list(it.accumulate(weights))
        i = random.random() * cumdist[-1]
        return choices[bisect.bisect(cumdist, i)]
    
except AttributeError: # no it.accumulate
    
    def sample_weighted_choice(choices):    # TODO check for bias
        """ Choose an element of choices (list of object, prob tuples). """
        #  Based on http://eli.thegreenplace.net/2010/01/22/weighted-random-generation-in-python/
        if isinstance(choices, dict):
            choices = list(choices.items())
        else:
            choices = list(choices)
        assert choices
        rnd = random.random() * sum(x[1] for x in choices)
        for c, w in choices:
            rnd -= w
            if rnd < 0:
                return c
        raise Exception("Shouldn't get here")

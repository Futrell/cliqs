from __future__ import division
import random
import math
from math import log, exp, sqrt
from collections import defaultdict
import itertools as it
import bisect
import functools

from rfutils.compat import *
from rfutils import singleton, unique, count, weighted_mean
import rfutils.entropy
try:
    import cytoolz as toolz
except ImportError:
    import toolz

from .prob import *

class ZeroSupportException(Exception):
    pass

INF = float('inf')

take = it.islice

expectation = weighted_mean


def rfunc_from(f):
    def decorator(rf_class):
        rf_class.sample = staticmethod(f)
        return singleton(rf_class)
    return decorator

def rapplicator(default_f):
    """ r applicator

    Decorator for function which describes a kind of function application for 
    rfuncs. For example, sample samples from an rfunc, score gets a score for
    some result, etc. 

    The name of the decorated function defines the method of the rfunc to be 
    called, and the function body defines the default behavior for non-random
    functions.

    """
    method_name = default_f.__name__
    def apply(f, *args, **kwds):
        if isinstance(f, rfunc):
            return getattr(f, method_name)(*args, **kwds)
        elif isinstance(f, functools.partial):
            if f.keywords:
                new_keywords = copy.copy(f.keywords)
                new_keywords.update(kwds)
                return apply(f.func, *(f.args + args), **new_keywords)
            else:
                return apply(f.func, *(f.args + args))
        else:
            return default_f(f, *args, **kwds)
    return apply

@rapplicator
def sample(f, *args, **kwds):
    return f(*args, **kwds)

@rapplicator
def mode(f, *args, **kwds):
    return f(*args, **kwds)

def score(f, result, *args, **kwds):
    # can't use rapplicator for this because it's a special case
    # requiring the results argument
    try:
        return f.score(result, *args, **kwds)
    except AttributeError:
        if isinstance(f, functools.partial):
            if f.keywords:
                new_keywords = copy.copy(f.keywords)
                new_keywords.update(kwds)
                return score(f.func, result, *(f.args + args), **new_keywords)
            else:
                return score(f.func, result, *(f.args + args))
        else:
            return safelog(f(*args, **kwds) == result)
    

@rapplicator
def enumeration(f, *args, **kwds):
    return [(f(*args, **kwds), 0)]

@rapplicator
def support(f, *args, **kwds):
    return [f(*args, **kwds)]

@rapplicator
def support_size(f, *args, **kwds):
    return 1

@rapplicator
def entropy(f, *args, **kwds):
    return 0

def reject(f, rf):
    while True:
        x = rf()
        if f(x):
            return x
    
class rfunc(object):
    """ random function

    A random function f draws a sample when called as f(x), provides the log
    probability of returning y when called as f.score(y, x), and returns
    an enumeration over possible values and their log probabilities when called
    as f.enumeration(x). Other methods are support and support_size.

    When it is not known that a function is random or not, then it is preferable
    to get the score (enumeration, etc.) as score(f, y, x), which will work
    for all pure functions as well as for random functions.

    """
    def sample(self, *args, **kwds):
        raise NotImplementedError

    def __call__(self, *args, **kwds):
        return self.sample(*args, **kwds)

    def score(self, result, *args, **kwds):
        raise NotImplementedError

    def support(self, *args, **kwds):
        raise NotImplementedError

    def enumeration(self, *args, **kwds):
        for item in self.support(*args, **kwds):
            yield item, self.score(item, *args, **kwds)

    def support_size(self, *args):
        return toolz.count(self.support(*args))

    def mode(self, *args, **kwds):
        top_sample, _ = max(self.enumeration(*args, **kwds), key=lambda x: x[-1])
        return top_sample

    def entropy(self, *args, **kwds):
        logps = (logp for _, logp in self.enumeration(*args, **kwds))
        return rfutils.entropy.entropy(map(exp, logps))
        

class conditional(rfunc):
    def __init__(self, rfs):
        self.rfs = rfs

    def sample(self, c):
        return self.rfs[c]()

    def score(self, result, c):
        return self.rfs[c].score(result)

    def enumeration(self, c):
        if c in self.rfs:
            return enumeration(self.rfs[c])
        else:
            return []

    def entropy(self, c):
        return entropy(self.rfs[c])

class product_rfunc(rfunc): 
    def __init__(self, *fs):
        self.fs = fs

    def sample(self, *args):
        return tuple(sample(f, *args) for f in self.fs)

    def score(self, result, *args):
        return sum(score(f, r, *args) for r, f in zip(result, self.fs))

    def enumeration(self, *args):
        for prod in it.product(*[enumeration(f, *args) for f in self.fs]):
            results, logps = zip(*prod)
            yield results, sum(logps)

def make_conditional(chooser, hists):
    distro = {context : rpartial(chooser, hist)
              for context, hist in hists.items()}
    distro = defaultdict(lambda: rpartial(chooser, {}),
                         distro)
    return conditional(distro)

class mixture(rfunc):
    def __init__(self, decide, consequences):
        self.decide = decide
        self.consequences = consequences
        
    def sample(self, *args, **kwds):
        decision = self.decide()
        return self.consequences[decision](*args)

    def score(self, result, *args, **kwds):
        return logsumexp(d_score + score(self.consequences[d], result, *args, **kwds)
                         for d, d_score in enumeration(self.decide))

    def support(self, *args, **kwds):
        supports = (c.support(*args, **kwds) for c in self.consequences.values())
        all_items = it.chain.from_iterable(supports)
        return unique(all_items)

    def entropy(self, *args, **kwds):
        def gen():
            for decision, d_score in enumeration(self.decide):
                yield d_score + entropy(self.consequences[decision], *args, **kwds)
        return logsumexp(gen())


def simple_mixture(weights):
    return mixture(
        rpartial(weighted_choice, weights),
        {model: model for model in weights.keys()},
    )
    
class rpartial(rfunc):
    def __init__(self, rf, *args):
        self.rf = rf
        self.args = args
        
    def sample(self, *more_args):
        return sample(self.rf, *(self.args + more_args))

    def score(self, result, *more_args):
        return score(self.rf, result, *(self.args + more_args))

    def enumeration(self, *more_args):
        return enumeration(self.rf, *(self.args + more_args))

    def support(self, *more_args):
        return support(self.rf, *(self.args + more_args))

    def support_size(self, *more_args):
        return support_size(self.rf, *(self.args + more_args))

    def entropy(self, *more_args):
        return entropy(self.rf, *(self.args + more_args))

def conditional_partial(f, d):
    return conditional({c: rpartial(f, x) for c, x in d.items()})

@singleton
class flip(rfunc):
    def sample(self, p):
        return random.random() < p

    def score(self, result, p):
        if result:
            return log(p)
        else:
            return log(1-p)

    def enumeration(self, p):
        return [(False, log(1 - p)), (True, log(p))]

    def entropy(self, p):
        return -log(p) - log(1-p)

    
def test_flip():
    import nose
    nose.tools.assert_almost_equal(score(flip, True, .25), log(.25))
    nose.tools.assert_almost_equal(score(flip, False, .25), log(.75))

    fp = functools.partial(flip, .25)
    assert sample_score_consistent(fp, True, .25, 1000)
    

@singleton
class choice(rfunc):
    def sample(self, choices):
        if isinstance(choices, dict):
            choices = list(choices.keys())
        else:
            choices = [x for x, _ in choices]
        return random.choice(choices)
            
    def score(self, choice, choices):
        if isinstance(choices, dict):
            choices = list(choices.keys())
        else:
            choices = [x for x, _ in choices]
        if not choices:
            raise ZeroSupportException
        if choice in choices:
            return -log(len(choices))
        else:
            return -INF

    def support(self, choices):
        if isinstance(choices, dict):
            return choices.keys()
        else:
            return (x for x, _ in choices)
    
    def entropy(self, choices):
        return log(count(choices))
    

@rfunc_from(sample_weighted_choice)
class weighted_choice(rfunc):
    def score(self, choice, choices):
        choices = dict(choices)
        if choice in choices and choices[choice] != 0:
            Z = sum(choices.values())
            return log(choices[choice]) - log(Z)
        else:
            return -INF

    def enumeration(self, choices):
        if isinstance(choices, dict):
            choices = list(choices.items())
        else:
            choices = list(choices)
        A = log(sum(weight for _, weight in choices))
        return [(choice, log(weight) - A) for choice, weight in choices]

    def support(self, choices):
        if isinstance(choices, dict):
            for choice in choices.keys():
                yield choice
        else:
            for choice, _ in choices:
                yield choice

    def entropy(self, choices):
        if isinstance(choices, dict):
            return rfutils.entropy.entropy(choices.values())
        else:
            return rfutils.entropy.entropy(value for _, value in choices)

def test_weighted_choice():
    import nose
    wc = weighted_choice
    d = {'a': .25, 'b': .25, 'c': .5}

    nose.tools.assert_almost_equal(.25, exp(wc.score('a', d)))
    nose.tools.assert_almost_equal(.25, exp(wc.score('b', d)))
    nose.tools.assert_almost_equal(.50, exp(wc.score('c', d)))

    wcp = functools.partial(weighted_choice, d)
    assert sample_score_consistent(wcp, 'a', .25, 1000)
    assert sample_score_consistent(wcp, 'b', .25, 1000)
    assert sample_score_consistent(wcp, 'c', .50, 1000)


@rfunc_from(iter)
class sample_until(rfunc):
    def score(self, xs, rf, sentinel):
        return sum(score(rf, x) for x in xs) + score(rf, sentinel)


@singleton
class shuffled(rfunc):
    def sample(self, xs):
        xs = list(xs)
        random.shuffle(xs)
        return xs
    
    def score(self, xs):
        return -math.lgamma(len(xs) + 1)

    def support(self, xs):
        return it.permutations(xs)

    def entropy(self, xs):
        return math.lgamma(len(xs) + 1)


def test_sample_until():
    import nose
    wcp = functools.partial(choice, {'a': 1, 'b': 1})
    assert all(x == 'a' for x in sample_until(wcp, 'b'))
    nose.tools.assert_almost_equal(score(sample_until, ['a'], wcp, 'b'),
                                   log(.25))
    nose.tools.assert_almost_equal(score(sample_until, ['a', 'a'], wcp, 'b'),
                                   log(.125))



def multinomial_ci(p, N, z=2.5):
    # from http://www.evanmiller.org/statistical-formulas-for-programmers.html
    # z=2.5 for 99% confidence interval
    z_squared = z ** 2
    sqrt_term = sqrt((p * (1 - p) + z_squared / (4*N))/N) 
    upper = (p + z_squared/(2*N) + z * sqrt_term) / (1 + z_squared/N)
    lower = (p + z_squared/(2*N) - z * sqrt_term) / (1 + z_squared/N)
    return lower, upper

def sample_score_consistent(rf, x, p, num_samples):
    samples = [rf() for _ in range(num_samples)]
    p_sampled = samples.count(x) / num_samples
    lower, upper = multinomial_ci(p, num_samples)
    return lower <= p_sampled <= upper

def scores_sum_to_one(rf, *args, **kwds):
    import nose
    Z = logsumexp(score(rf, x, *args, **kwds) for x in support(rf, *args, **kwds))
    nose.tools.assert_almost_equal(Z, 0)

def enumeration_sums_to_one(rf, *args, **kwds):
    import nose
    Z = logsumexp(score for _, score in enumeration(rf, *args, **kwds))
    nose.tools.assert_almost_equal(Z, 0)

def samples_above_p_threshold(rf, p_thresh=.05):
    while True:
        x = rf()
        p = exp(score(rf, x))
        if p >= p_thresh:
            yield x, p
            
if __name__ == '__main__':
    import nose
    nose.runmodule()    

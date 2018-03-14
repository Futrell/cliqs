#!/usr/bin/python3
import sys
import random
import itertools as it
from collections import Counter, defaultdict
from math import log, exp
import functools
import math
import operator
import copy

from pyrsistent import pbag
import rfutils
from rfutils.compat import *
import rfutils.ordering as o

from .rfunc import *
from . import conditioning as cond
from . import depgraph



VERBOSE = 1

HALT = '!HALT!'
ROOT = 0

K = 1/100

# SAME_LIMIT is the maximum number of equal values allowed in an array whose
# permutations we will be computing, e.g. an array of values of dependents of
# a head. Without this, we run into slowness from exponential time complexity.
# In the case of filtering dependents, it seems that this mostly serves to filter
# out lists and foreign expressions, which we don't want anyway.
SAME_LIMIT = 20
PHRASE_LIMIT = 20

class BadSentenceException(Exception):
    pass

class SameLimitException(BadSentenceException):
    pass

class CounterWithSum(Counter):
    @rfutils.lazy_property
    def sum(self):
        return sum(self.values())

def same_limit_ok(xs):
    return all(c <= SAME_LIMIT for c in Counter(xs).values())

def check_same_limit_ok(xs):
    if not same_limit_ok(xs):
        raise SameLimitException

def sentence_same_limit_ok(cond, s):
    return all(same_limit_ok(cond(s, depgraph.immediate_phrase_of(s, n), n))
               for n in s.nodes())

@rfutils.singleton    
class randlin(rfunc):
    def sample(self, sentence, phrase, h):
        indices = list(range(len(phrase)))
        random.shuffle(indices)
        return indices

    def score(self, order, sentence, phrase, h):
        return -math.lgamma(len(phrase) + 1)

    def support(self, sentence, phrase, h):
        indices = list(range(len(phrase)))
        return it.permutations(indices)

    def entropy(self, sentence, phrase, h):
        return math.lgamma(len(phrase) + 1)

    
# design space for linearizations.
# deptype * dpos * hpos * pos2/pos * CH

class proj_phrase_order_rfunc(rfunc):
    ordering = o.sorted_ordering
    
class EisnerModelC(rfunc):
    def __init__(self, conditioning_fn, left, right, chooser=weighted_choice):
        self.chooser = chooser
        self.conditioning_fn = conditioning_fn
        
        self.left_hists = {k: CounterWithSum(v) for k, v in left.items()}
        self.right_hists = {k: CounterWithSum(v) for k, v in right.items()}

        self.left_word_distro = conditional_partial(self.chooser, self.left_hists)
        self.right_word_distro = conditional_partial(self.chooser, self.right_hists)

        def f(hist):
            return sample_until(rpartial(self.chooser, hist), HALT)
            
        self.left_distro = conditional_partial(f, self.left_hists)
        self.right_distro = conditional_partial(f, self.right_hists)

    @classmethod
    def from_sentences(cls, conditioning_fn, sentences, chooser=weighted_choice):
        left = defaultdict(Counter)
        right = defaultdict(Counter)
        for sentence in sentences:
            for n in sentence.nodes():
                phrase = conditioning_fn(sentence, depgraph.immediate_phrase_of(sentence, n), n)
                h_index = cond.head_index(phrase)
                head = phrase[h_index]
                left_of_head = phrase[:h_index]
                right_of_head = phrase[(h_index+1):]
                left[head] += Counter(left_of_head)
                left[head][HALT] += 1
                right[head] += Counter(right_of_head)
                right[head][HALT] += 1
        return cls(conditioning_fn, left, right, chooser=chooser)

    def sample(self):
        return self.sample_from(cond.ROOT_HEAD)

    def sample_from(self, node):
        wrong
        left = self.left_distro(node)
        right = self.right_distro(node)
        return [
            list(map(self.sample_from, left)),
            node,
            list(map(self.sample_from, right)),
        ]

    def score(self, sentence):
        wrong
        left, head, right = sentence
        left_score = self.left_distro.score(left, head)
        right_score = self.right_distro.score(right, head)
        left_rec = sum(map(self.score, left))
        right_rec = sum(map(self.score, right))
        return left_score + right_score + left_rec + right_rec

    
class DepNGram(rfunc):
    def __init__(self, conditioning_fn,
                 n,
                 left_transitions,
                 right_transitions,
                 chooser=weighted_choice):
        self.chooser = chooser
        self.left_transitions = left_transitions
        self.right_transitions = right_transitions
        self.left_distro = conditional({head: make_conditional(self.chooser, transitions)
                                        for head, transitions in left_transitions.items()})
        self.right_distro = conditional({head: make_conditional(self.chooser, transitions)
                                        for head, transitions in right_transitions.items()})        
        self.conditioning_fn = conditioning_fn
        self.n = n
        # make ngram distros
        
    @classmethod
    def from_sentences(cls, conditioning_fn, n, sentences, chooser=weighted_choice):
        left_transitions = dict()
        right_transitions = dict()
        for sentence in sentences:
            for node in sentence.nodes():
                phrase = depgraph.immediate_phrase_of(sentence, node)
                conditioned_phrase = conditioning_fn(sentence, phrase, node)
                h = cond.head_index(conditioned_phrase)
                head, (left, right) = pop(conditioned_phrase, h)
                left = reversed(left)
                for ngram in ngrams(left, n):
                    context = ngram[:-1]
                    thing = ngram[-1]
                    context = tuple(context)
                    if head not in left_transitions:
                        left_transitions[head] = dict()
                    if context not in left_transitions[head]:
                        left_transitions[head][context] = CounterWithSum()
                    left_transitions[head][context][thing] += 1

                for ngram in ngrams(right, n):
                    context = ngram[:-1]
                    thing = ngram[-1]
                    context = tuple(context)
                    if head not in right_transitions:
                        right_transitions[head] = dict()
                    if context not in right_transitions[head]:
                        right_transitions[head][context] = CounterWithSum()
                    right_transitions[head][context][thing] += 1

        return cls(conditioning_fn,
                   n,
                   left_transitions,
                   right_transitions,
                   chooser=chooser,
                   )

def lazy_weighted_choice(choices):
    total = 0
    winner = 0
    for choice, weight in choices:
        total += weight
        if random.random() * total < weight:
            winner = choice
    return winner

class EisnerModelCOrders(proj_phrase_order_rfunc):
    def __init__(self, depgen):
        self.depgen = depgen
        Z_left = Counter({n: sum(c for x,c in cc.items() if x != HALT)
                          for n, cc in self.depgen.left_hists.items()})
        Z_right = Counter({n: sum(c for x,c in cc.items() if x != HALT)
                           for n, cc in self.depgen.right_hists.items()})
        Z = {n: Z_left[n] + Z_right[n] for n in set(Z_left.keys()).union(Z_right.keys())}
        self.left_prior = conditional({n: rpartial(flip, Z_left[n] / Z[n]) for n in Z if Z[n]})

    def sample(self, sentence, phrase, h):
        if len(phrase) > PHRASE_LIMIT:
            raise BadSentenceException
        old_sequence = self.depgen.conditioning_fn(sentence, phrase, h)
        h = cond.head_index(old_sequence)
        head, (left, right) = pop(old_sequence, h)
        deps = left + right

        # need to choose a partition without loading all of them into memory.
        # this can be done at the expense of time...
        # the implementation below enumerates the partitions exactly once,
        # at the cost of using a slower weighted_choice function
        the_partitions = rfutils.zipmap(lambda parts: exp(self.score_parts(head, *parts)),
                                        rfutils.partitions(deps))
        left, right = lazy_weighted_choice(the_partitions)
        left = list(left)
        right = list(right)
        random.shuffle(left)
        random.shuffle(right)
        new_sequence = list(it.chain(left, [head], right))
        return o.sample_indices_in(old_sequence, new_sequence)
                                    
    def score(self, indices, sentence, phrase, h):
        assert len(indices) == len(phrase)
        # do some correction for same indices here? no; if we did then to calculate
        # a score we'd have to sum all the equivalent indices.
        conditioned_phrase = self.depgen.conditioning_fn(sentence, phrase, h)
        new_conditioned_phrase = list(o.reorder(conditioned_phrase, indices))
        num_options = product(math.factorial(c) for c in Counter(new_conditioned_phrase).values())
        return self.score_seq(new_conditioned_phrase) - log(num_options)

    def score_seq(self, sequence):
        sequence = list(sequence)
        h = cond.head_index(sequence)
        head = sequence.pop(h)
        parts = self.split_parts(sequence, h)
        parts_score = self.score_parts(head, *parts)
        if math.isinf(parts_score):
            return -INF # otherwise we get nan down the line
        norm = self.norm(head, parts)
        return parts_score - norm

    def split_parts(self, sequence, h):
        return sequence[:h], sequence[h:]

    def norm(self, head, parts):
        deps = list(flat(parts))
        return logsumexp(self.score_parts(head, *parts)
                         - self.score_order(head, *parts) # minus!?
                         for parts in rfutils.partitions(deps))

    def score_order(self, head, left, right):
        # probability of this order given the partition
        return -math.lgamma(len(left) + 1) - math.lgamma(len(right) + 1)
    
    def score_parts(self, head, left, right):
        l_score = sum(self.depgen.left_word_distro.score(w, head)
                      for w in left)
        r_score = sum(self.depgen.right_word_distro.score(w, head)
                      for w in right)
        return l_score + r_score

def product(xs):
    result = 1
    for x in xs:
        result *= x
    return result

def pop(seq, i):
    seq = list(seq)
    return seq[i], (seq[:i], seq[(i+1):])

def gen_trace(gen):
    def wrapper(*a, **k):
        result = list(gen(*a, **k))
        rfutils.err("%s(%s) = %s" % (gen.__name__,
                                     ", ".join(map(str, a)),
                                     str(result)))
        return result
    return wrapper

class DepNGramOrders(EisnerModelCOrders):
    def __init__(self, depgen):
        self.depgen = depgen

    def sample(self, sentence, phrase, h):
        # use dynamic programming to find the probability of a given context C
        # and generated multiset A:
        #
        # p(a | C, A) = ( p(a|C) + Z_{A-{a} | trunc(C, N-1)} ) / Z_{A | C}
        #
        sequence = self.depgen.conditioning_fn(sentence, phrase, h)
        h_index = cond.head_index(sequence)
        head, (l_deps, r_deps) = pop(sequence, h_index)
        deps = l_deps + r_deps

        def gen_left_probs(context, xs_bag):
            left_score = self.depgen.left_distro.rfs[head].score
            right_score = self.depgen.right_distro.rfs[head].score
            n = self.depgen.n
            yield HALT, exp(left_score(HALT, context)
                            + ngram_perm_norm_from((right_score, n, context, xs_bag)))
            for x, rest in rfutils.thing_and_rest(xs_bag): # potential inefficiency
                next_context = truncate(context + (x,), n - 1)
                A_rest = double_ngram_left_perm_norm_from((left_score,
                                                          right_score,
                                                          n,
                                                          next_context,
                                                          pbag(rest))) 
                yield x, exp(left_score(x, context) + A_rest)

        def gen_right_probs(context, xs_bag):
            score = self.depgen.right_distro.rfs[head].score
            n = self.depgen.n
            if not xs_bag:
                yield HALT, exp(score(HALT, context))
            else:
                for x, rest in rfutils.thing_and_rest(xs_bag): # potential inefficiency
                    next_context = truncate(context + (x,), n - 1)
                    A_rest = ngram_perm_norm_from((score, n, next_context, pbag(rest))) 
                    yield x, exp(score(x, context) + A_rest)

        def sample_part(get_probs, context, xs_bag):
            while xs_bag:
                probs = get_probs(context, xs_bag)
                x = weighted_choice(list(probs))
                if x is HALT:
                    break
                else:
                    yield x
                    context = truncate(context + (x,), len(context))
                    xs_bag = xs_bag.remove(x)

        deps_bag = pbag(deps)
        start = (HALT,) * (self.depgen.n - 1)
        left = list(sample_part(gen_left_probs, start, deps_bag))
        remainder = deps_bag - pbag(left)
        right = sample_part(gen_right_probs, start, remainder)

        new_sequence = list(it.chain(reversed(left), [head], right))

        rfutils.get_cache(double_ngram_left_perm_norm_from).clear()
        rfutils.get_cache(ngram_perm_norm_from).clear()
        
        return o.sample_indices_in(sequence, new_sequence)
        
    def score_seq(self, sequence):
        sequence = list(sequence)
        h = cond.head_index(sequence)
        head = sequence.pop(h)
        left, right = self.split_parts(sequence, h)
        left = list(reversed(left))
        left_rf = self.depgen.left_distro.rfs[head]
        left_score = ngram_sequence.score(left, left_rf, self.depgen.n)
        if math.isinf(left_score):
            return -INF # avoid nan
        right_rf = self.depgen.right_distro.rfs[head]
        right_score = ngram_sequence.score(right, right_rf, self.depgen.n)
        if math.isinf(right_score):
            return -INF # avoid nan
        norm = double_ngram_perm_norm(left_rf.score,
                                      right_rf.score,
                                      self.depgen.n,
                                      left + right)
        return left_score + right_score - norm


def truncate(xs, n):
    """ Truncate a sequence to length n by deleting elements from the front. """
    if n == 0:
        return xs * 0 # empty sequence of the right type
    elif n > 0:
        return xs[-n:]
    else:
        raise ValueError("Cannot truncate to length %s" % n)

def double_ngram_perm_norm(score_left, score_right, n, xs): # xs :: Iterable
    start = (HALT,) * (n - 1)
    result = double_ngram_left_perm_norm_from((score_left,
                                            score_right,
                                            n,
                                            start,
                                            pbag(xs)))
    rfutils.get_cache(double_ngram_left_perm_norm_from).clear()
    return result


@rfutils.fast_memoize
def double_ngram_left_perm_norm_from(t):
    score_left, score_right, n, context, xs = t  # xs :: PBag
    def gen():
        start = (HALT,) * (n - 1)
        yield score_left(HALT, context) + ngram_perm_norm_from((score_right,
                                                               n,
                                                               start,
                                                               xs))
        for thing, num_thing in xs._counts.iteritems():
            rest = xs.remove(thing)
            next_context = truncate(context + (thing,), n - 1)
            yield (score_left(thing, context) + log(num_thing)
                   + double_ngram_left_perm_norm_from((score_left,
                                                      score_right,
                                                      n,
                                                      next_context,
                                                      rest)))
    return logsumexp(gen())


def ngram_perm_norm(score, n, xs):
    """ N-gram Permutation Norm

    Get the log norm of N-gram model scores of permutations of xs.

    """
    # dynamic programming solution
    start = (HALT,) * (n - 1)
    result = ngram_perm_norm_from((score, n, start, pbag(xs)))
    rfutils.get_cache(ngram_perm_norm_from).clear()
    return result

@rfutils.fast_memoize
def ngram_perm_norm_from(t):
    # Suppose we have xs == aaab. Then the possible
    # permutations are those equivalent to aaab, aaba, abaa, baaa.
    # For example, aaab -> a1..., a2..., a3...
    # So each permutation starting in a is repeated num(a) times.
    score, n, context, xs = t # xs :: PBag
    def gen():
        for thing, num_thing in xs._counts.iteritems():
            rest = xs.remove(thing)
            next_context = truncate(context + (thing,), n - 1)
            yield (score(thing, context) + log(num_thing)
                   + ngram_perm_norm_from((score, n, next_context, rest)))
    if xs:
        return logsumexp(gen())
    else:
        return score(HALT, context)

def ngrams(xs, n):
    assert n >= 1
    buildup = [HALT] * (n - 1)
    for i, x in enumerate(xs):
        buildup.append(x)
        yield tuple(truncate(buildup, n))
    buildup.append(HALT)
    yield tuple(truncate(buildup, n))

@rfutils.singleton
class ngram_sequence(rfunc):
    def sample(self, f, n):
        context = (HALT,) * (n - 1)
        while True:
            x = f(context)
            if x is HALT:
                break
            else:
                yield x
                context = truncate(context + (x,), n - 1)
        
    def score(self, sequence, f, n):
        def gen():
            for ngram in ngrams(sequence, n):
                yield score(f, ngram[-1], ngram[:-1])
        return sum(gen())

def score_ngram_sequence(score, n, sequence):
    def gen():
        for ngram in ngrams(sequence, n):
            yield score(ngram[-1], ngram[:-1])
    return sum(gen())

class ObservedOrders(proj_phrase_order_rfunc):
    def __init__(self, conditioning_fn, orders=None):
        self.conditioning_fn = conditioning_fn
        if orders is None:
            self.orders = {}
        else:
            self.orders = {k : CounterWithSum(v) for k, v in orders.items()}

    @classmethod
    def from_sentences(cls, conditioning_fn, sentences):
        def conditioned_phrases():
            for sentence in sentences:
                for h in sentence.nodes():
                    phrase = depgraph.immediate_phrase_of(sentence, h)
                    conditioned_phrase = conditioning_fn(sentence, phrase, h)
                    if same_limit_ok(conditioned_phrase):
                        yield conditioned_phrase
                    elif VERBOSE > 0:
                        rfutils.err("Bad phrase in sentence at line %s: %s"
                            % (sentence.start_line, conditioned_phrase))
                        
        orders = cls.ordering.count_orders(conditioned_phrases())
        return cls(conditioning_fn, orders)

    def order_distro(self, sentence, phrase, h):
        conditioned_phrase = self.conditioning_fn(sentence, phrase, h)
        return self.orders[self.ordering.canonical_order(conditioned_phrase)]

    def sample(self, f, sentence, phrase, h):
        conditioned_phrase = self.conditioning_fn(sentence, phrase, h)
        canonical = self.ordering.canonical_order(conditioned_phrase)
        distro = self.orders.get(canonical,
                                 CounterWithSum({tuple(range(len(canonical))): 0}))
        indices = f(distro) # a reordering of the canonical order
        new_conditioned_phrase = o.reorder(canonical, indices)
        ret = o.sample_indices_in(conditioned_phrase, new_conditioned_phrase)
        assert len(ret) == len(phrase)
        return ret

    def score(self, indices, f, sentence, phrase, h):
        conditioned_phrase = self.conditioning_fn(sentence, phrase, h)
        new_conditioned_phrase = list(o.reorder(conditioned_phrase, indices))
        # in principle, shouldn't need to do this any more, since the cost
        # of large number of indices_in is in time, not in space.
        check_same_limit_ok(new_conditioned_phrase)
        num_options = o.num_equivalent_permutations(new_conditioned_phrase)
        canonical = self.ordering.canonical_order(new_conditioned_phrase)
        distro = self.orders.get(canonical,
                                 CounterWithSum({tuple(range(len(canonical))): 0}))

        new_canonical_indices = next(self.ordering.indices_in_canonical_order(new_conditioned_phrase))
        for distro_indices in distro:
            if self.ordering.permutations_equivalent(canonical, new_canonical_indices, distro_indices):
                return score(f, distro_indices, distro) - log(num_options)
        return score(f, distro_indices, distro) - log(num_options)

    def support(self, f, sentence, phrase, h):
        conditioned_phrase = self.conditioning_fn(sentence, phrase, h)
        canonical = self.ordering.canonical_order(conditioned_phrase)
        distro = self.orders.get(canonical,
                                 CounterWithSum({tuple(range(len(canonical))): 0}))
        indices_into_canonical = support(f, distro)
        for indices in indices_into_canonical:
            new_conditioned_phrase = o.reorder(canonical, indices)
            ret = o.sample_indices_in(conditioned_phrase, new_conditioned_phrase)
            assert len(ret) == len(phrase)
            yield ret
        
        #options = self.ordering.indices_in_canonical_order(new_conditioned_phrase)
        #for option in options:
        #    if option in distro:
        #        assert len(option) == len(phrase)
        #        return score(f, option, distro) - log(num_options)
        #assert len(option) == len(phrase)
        #return score(f, option, distro) - log(num_options)

class randlin_oo(proj_phrase_order_rfunc):
    def __init__(self, oo, chooser):
        self.oo = oo
        self.chooser = chooser

    def sample(self, sentence, phrase, h):
        return self.oo(self.chooser, sentence, phrase, h)

    def score(self, indices, sentence, phrase, h):
        assert len(indices) == len(phrase)
        return score(self.oo, indices, self.chooser, sentence, phrase, h)

    def support(self, sentence, phrase, h):
        return support(self.oo, self.chooser, sentence, phrase, h)

def randlin_licit(oo):
    return randlin_oo(oo, choice)

def randlin_mle(oo):
    return randlin_oo(oo, weighted_choice)

class permutation_add_k(rfunc):
    def __init__(self, k, chooser):
        self.k = k
        self.chooser = chooser
        
    def sample(self, distro):
        indices = rfutils.first(distro.keys()) # get an arbitrary list of indices
        Z = distro.sum if distro else 0
        N = self.k * math.factorial(len(indices))
        # decide if we'll go to the flat distro or not
        if flip(Z / (Z + N)):
            # should never get here if Z == 0
            return self.chooser(distro)
        else:
            result = list(indices)
            random.shuffle(result)
            return result

    def score(self, result, distro):
        assert len(result) == len(rfutils.first(distro.keys()))
        base_score = score(self.chooser, result, distro)
        flat_score = -math.lgamma(len(result) + 1)

        num_base_obs = distro.sum if distro else 0
        num_flat_obs = self.k * math.factorial(len(result))
        
        p_base = num_base_obs / (num_base_obs + num_flat_obs)
        logp_base = safelog(p_base)
        logp_flat = log(1 - p_base)
        return logaddexp(logp_base + base_score, logp_flat + flat_score)

def all_deps(conditioning_fn, sentences):
    def gen():
        for s in sentences:
            for n in s.nodes():
                for item in conditioning_fn(s, depgraph.immediate_phrase_of(s, n), n):
                    if not cond.is_head(item):
                        yield item
    return set(gen())

class add_k(rfunc):
    # in a faster world, implement this as a mixture
    
    def __init__(self, k, support):
        self.k = k
        self._support = list(support)

    def support(self, distro):
        return self._support

    def sample(self, distro):
        Z = distro.sum if distro else 0
        N = len(distro)
        if flip(Z / (Z + self.k * len(self._support))):
            return weighted_choice(distro)
        else:
            return choice(self._support)

    def score(self, result, distro):
        num_base_obs = distro.sum if distro else 0
        # in a faster world, we do: base_score = weighted_choice.score(result, distro)
        if result in distro and distro[result] > 0:
            base_score = log(distro[result]) - log(num_base_obs)
        else:
            base_score = -INF

        N = len(self._support)
        num_flat_obs = self.k * N
        flat_score = -log(N)
        
        p_base = num_base_obs / (num_base_obs + num_flat_obs)
        logp_base = safelog(p_base)
        logp_flat = safelog(1 - p_base)
        return logaddexp(logp_base + base_score, logp_flat + flat_score)

    def entropy(self, distro):
        Z = distro.sum if distro else 0
        N = len(distro)
        p_base = Z / (Z + self.k * len(self._support))
        return (p_base * weighted_choice.entropy(distro)
                + (1-p_base) * choice.entropy(self._support))
        
class randlin_add_k(randlin_oo):
    def __init__(self, oo, k):
        self.oo = oo
        self.chooser = permutation_add_k(k, weighted_choice)

@rfutils.singleton
class proj_lin_from(rfunc):
    def sample(self, f, sentence, h):    
        """ projective linearize from

        Yield nodes representing a projective linearization of the subtree of
        sentence rooted at node h, produced by applying the function f to h and 
        its dependents and then applying proj_lin_from to its dependents
        recursively.
    
        """
        phrase = depgraph.immediate_phrase_of(sentence, h)
        phrase_order_indices = f(sentence, phrase, h)
        return flat(proj_lin_from(f, sentence, n) if n != h else [n]
                    for n in o.reorder(phrase, phrase_order_indices))

    def score(self, lin, f, sentence, h):
        phrase = depgraph.immediate_phrase_of(sentence, h)
        phrase_set = set(phrase)
        relevant_lin = [n for n in lin if n in phrase_set]
        # this indices_in should be unique:
        phrase_order_indices = rfutils.the_only(o.indices_in(phrase, relevant_lin))
        the_score = score(f, phrase_order_indices, sentence, phrase, h)
        return the_score + sum(proj_lin_from.score(lin, f, sentence, n)
                               for n in phrase if n != h)

    def mode(self, f, sentence, h):
        phrase = depgraph.immediate_phrase_of(sentence, h)
        phrase_order_indices = mode(f, sentence, phrase, h)
        return flat(proj_lin_from.mode(f, sentence, n) if n != h else [n]
                    for n in o.reorder(phrase, phrase_order_indices))

    def support(self, f, sentence, h): 
        phrase = depgraph.immediate_phrase_of(sentence, h)
        perms = support(f, sentence, phrase, h)
        for perm in perms:
            new_phrase = o.reorder(phrase, perm)
            # supports example: [[[1, 2], [2, 1]], [[3]], [[4, 5], [5, 4]]]
            supports = [
                proj_lin_from.support(f, sentence, n) if n != h else [[n]]
                for n in new_phrase
            ]
            # subtrees example: [[[1, 2], [3], [4, 5]], [[2, 1], [3], [4, 5]], ...]
            subtrees = it.product(*supports)
            for subtree in subtrees:
                # subtree example: [[1, 2], [3], [4, 5]]
                yield tuple(flat(subtree))

flat = it.chain.from_iterable

@rfutils.singleton
class proj_lin(rfunc):
    """ projective linearize
    
    Return a projective linearization (tuple of nodes) resulting from
    applying the function f to each head node.
    
    f must be a function of a sentence, phrase, and the phrase's head node which 
    returns a linearization (sequence of indices) of that node and its 
    immediate dependents.
    
    """
    def sample(self, f, sentence):
        return tuple(proj_lin_from(f, sentence, ROOT))

    def mode(self, f, sentence):
        return tuple(proj_lin_from.mode(f, sentence, ROOT))

    def score(self, lin, f, sentence):
        return proj_lin_from.score(lin, f, sentence, ROOT)

    def support(self, f, sentence):
        return proj_lin_from.support(f, sentence, ROOT)


class WeightedLin(object):
    def __init__(self, conditioning_fn, weights):
        self.conditioning_fn = conditioning_fn # :: a -> b
        self.weights = weights # :: b -> Num

    def weight_of(self, w):
        if cond.is_head(w):
            return 0
        else:
            return self.weights[w]

    def __call__(self, sentence, phrase, h):
        conditioned_phrase = self.conditioning_fn(sentence, phrase, h)
        ordered_phrase = sorted(conditioned_phrase, key=self.weight_of)
        return o.sample_indices_in(conditioned_phrase, ordered_phrase)


def true_ngram_perm_norm(score, n, xs):
    return logsumexp(score_ngram_sequence(score, n, perm) for perm in it.permutations(xs))

def true_double_ngram_perm_norm(score_left, score_right, n, xs):
    def gen():
        for perm in it.permutations(xs):
            for i in range(len(perm) + 1):
                left, right = perm[:i], perm[i:]
                left_score = score_ngram_sequence(score_left, n, left)
                right_score = score_ngram_sequence(score_right, n, right)
                yield left_score + right_score
    return logsumexp(gen())

def test_ngram_perm_norm():
    def score(x, context):
        if context == ('a',):
            return {'a': 7/10, 'b': 2/10, HALT: 1/10}[x]
        elif context == ('b',):
            return {'a': 5/10, 'b': 4/10, HALT: 1/10}[x]
        elif context == (HALT,):
            return {'a': 7/10, 'b': 1/10, HALT: 2/10}[x]
        else:
            raise ValueError

    tests = [
        "a",
        "ab",
        "abbbbb",
        "aaaabbbb",
    ]

    def is_close(x, y):
        return (x - y) < 0.000001

    for test in tests:
        assert is_close(true_ngram_perm_norm(score, 2, test),
                        ngram_perm_norm(score, 2, test))

def test_double_ngram_perm_norm():
    def score(x, context):
        if context == ('a',):
            return {'a': 7/10, 'b': 2/10, HALT: 1/10}[x]
        elif context == ('b',):
            return {'a': 5/10, 'b': 4/10, HALT: 1/10}[x]
        elif context == (HALT,):
            return {'a': 7/10, 'b': 1/10, HALT: 2/10}[x]
        else:
            raise ValueError    

    tests = [
        "a",
        "ab",
        "abbbbb",
        "aaaabbbb",
    ]

    def is_close(x, y):
        return (x - y) < 0.000001

    for test in tests:
        assert is_close(true_double_ngram_perm_norm(score, score, 2, test),
                        double_ngram_perm_norm(score, score, 2, test))




if __name__ == '__main__':
    import nose
    nose.runmodule()


    


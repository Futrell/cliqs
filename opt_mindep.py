from __future__ import division, print_function
from collections import defaultdict, Counter
import sys
import random
import time
import itertools

import networkx as nx
from rfutils import mreduce_by_key

import conditioning as cond
import mindep

WEIGHT_MIN = -1
WEIGHT_MAX = 1

VERBOSE = 1

INFINITY = float('inf')

# functions sentence x (head id, dep id, dt dict) -> something

def get_deptype(sentence, h_d_dt):
    _, _, dt = h_d_dt
    return dt['deptype']

def get_deppos(sentence, h_d_dt):
    _, d, _ = h_d_dt
    return sentence.node[d].get('pos')

def get_headpos(sentence, h_d_dt):
    h, _, _ = h_d_dt
    return sentence.node[h].get('pos')

def product_f(*fs):
    return lambda *args: tuple(f(*args) for f in fs)

get_deptype_and_headpos = product_f(get_deptype, get_headpos)

DEFAULT_THING_FN = product_f(get_headpos, get_deptype)

def rand_fixed_weights(deptypes, head_final=False):
    if head_final:
        return {dt : rand_in_interval(.01, WEIGHT_MAX) for dt in deptypes}
    else:
        return {dt : rand_in_interval(WEIGHT_MIN, WEIGHT_MAX) for dt in deptypes}

def randlin_from_weights(sentence, weights, thing_fn):
    linearization = get_linearization(sentence, weights, thing_fn=thing_fn)
    cost = mindep.deplen(sentence, linearization)
    return cost, linearization    

def randlin_fixed_weights(sentence, thing_fn=DEFAULT_THING_FN, head_final=False):
    deptypes = {thing_fn(sentence, edge) for edge in sentence.edges_iter(data=True)}
    weights = rand_fixed_weights(deptypes, head_final=head_final)
    return randlin_from_weights(sentence, weights, thing_fn)

def get_total_dependency_length(corpus, weights, thing_fn=DEFAULT_THING_FN):
    """ Get the total dependency length in a corpus, when linearized 
    deterministically by given deptype weights. """
    total = 0
    for sentence in corpus.sentences():
        linearization = get_linearization(sentence, weights, thing_fn=thing_fn)
        deplen = mindep.deplen(sentence, linearization)
        total += deplen
    return total

def test_set_surprisal(corpus, weights, thing_fn=DEFAULT_THING_FN, n=5):
    import kenlmwrapper
    def reorder_sentence(s):
        linearization = get_linearization(s, weights, thing_fn=thing_fn)
        return [s.node[n]['word'] for n in linearization if n != 0]
                    
    tic = time.clock()
    sentences = list(map(reorder_sentence, corpus.sentences()))
    N = len(sentences)
    train_set_size = N - (N // 10)
    train_set = sentences[:train_set_size]
    test_set = sentences[(train_set_size+1):]
    model = kenlmwrapper.make_kenlm(train_set, n, verbose=False)
    result = -sum(model.score(" ".join(s)) for s in test_set)
    toc = time.clock()
    if VERBOSE:
        print("Time=%s" % (toc - tic), file=sys.stderr)
        print("Test set surprisal=%s" % result, file=sys.stderr)
    return result

def linear_mixture(f1, f2, w):
    def mix(*a, **k):
        return w * f1(*a, **k) + (1 - w)*f2(*a, **k)
    return mix

def mean(xs):
    total = 0
    n = 0
    for x in xs:
        total += x
        n += 1
    return total/n

def lists_by_key(xs):
    return mreduce_by_key(list.append, xs, list)

def find_optimal_deplen_weights_head_final(corpus,
                                           thing_fn=DEFAULT_THING_FN):
    deptypes = get_dependency_types(corpus, thing_fn=thing_fn)
    def gen():
        for s in corpus.sentences():
            for edge in s.edges(data=True):
                _, d, _ = edge
                yield thing_fn(s, edge), s.num_words_in_phrase(d)
    weights_by_deptype = lists_by_key(gen())
    return {
        thing : -mean(weights) for thing, weights in weights_by_deptype.items()
    }

def linear_project_into(xs, new_low, new_high):
    assert new_low <= new_high
    xs = list(xs)
    old_low = min(xs)
    old_high = max(xs)
    old_span = old_high - old_low
    new_span = new_high - new_low
    return [
        new_span * (x - old_low)/old_span + new_low
        for x in xs
    ]

def find_optimal_weights(corpus,
                         epsilon=0,
                         thing_fn=DEFAULT_THING_FN,
                         objective_fn=get_total_dependency_length,
                         num_initial_restarts=1):
    deptypes = get_dependency_types(corpus, thing_fn=thing_fn)
    if VERBOSE:
        print("Found deptypes: %s" % " ".join(map(str, deptypes)), file=sys.stderr)

    def initial_weights():
        for _ in range(num_initial_restarts):
            weights = {dt : rand_in_interval(WEIGHT_MIN, WEIGHT_MAX) for dt in deptypes}
            yield objective_fn(corpus, weights, thing_fn=thing_fn), weights

    initial_objective, weights = min(initial_weights())

    if VERBOSE:
        print("After %s random initializations," % num_initial_restarts, file=sys.stderr)
        print("optimizing from initial objective: %s" % initial_objective, file=sys.stderr)
    
    interaction_table = get_interaction_table(corpus, thing_fn=thing_fn)
    iteration = 0
    old_score = initial_objective
    while True:
        try:
            if VERBOSE:
                print("Running sweep %s" % iteration, file=sys.stderr)
            score = find_optimal_weights_sweep(corpus,
                                               weights,
                                               interaction_table,
                                               thing_fn=thing_fn,
                                               objective_fn=objective_fn)
            assert score <= old_score
            print(score, file=sys.stderr)
            if old_score - score <= epsilon:
                break
            old_score = score
            iteration += 1
        except KeyboardInterrupt as e:
            if __name__ == '__main__':
                try:
                    return score, weights
                except UnboundLocalError: # score not defined yet
                    score = objective_fn(
                        corpus,
                        weights,
                        thing_fn=thing_fn
                    )
                    return score, weights
            else:
                raise e
    return score, weights

def rand_in_interval(low, high):
    return (high - low) * random.random() + low

def random_segments(items, maxlen=1):
    assert maxlen < 2
    return ((item,) for item in items)

def get_dependency_types(corpus, thing_fn=DEFAULT_THING_FN):
    return set(thing_fn(sentence, edge) for sentence in corpus.sentences()
               for edge in sentence.edges_iter(data=True))

def find_optimal_weights_sweep(corpus,
                               weights, # TO BE MUTATED!
                               interaction_table,
                               thing_fn=DEFAULT_THING_FN,
                               objective_fn=get_total_dependency_length,
                               max_to_resample=1):
    old_objective = objective_fn(
        corpus,
        weights,
        thing_fn=thing_fn,
    )
    dependency_types = list(weights.keys())
    random.shuffle(dependency_types)
    segments = random_segments(dependency_types, maxlen=max_to_resample)
    for dependency_type_group in segments:
        old_group_weights = [weights[t] for t in dependency_type_group]
        interacting_types = set().union(*[interaction_table[t]
                                          for t in dependency_type_group])
        
        # here, [0] is the weight for the head:
        ordered_weights = sorted([weights[t] for t in interacting_types] + [0])
        proposed_group_weights = []
        tic = time.clock()
        the_intervals = list(intervals(ordered_weights, WEIGHT_MIN, WEIGHT_MAX))
        n = len(dependency_type_group)
        interval_groups = itertools.product(*[the_intervals] * n)
        for interval_group in interval_groups:
            new_weights = []
            for t, (low, high) in zip(dependency_type_group, interval_group):
                new_weight = rand_in_interval(low, high)
                weights[t] = new_weight
                new_weights.append(new_weight)
            result = objective_fn(
                corpus,
                weights,
                thing_fn=thing_fn
            )
            for t, old_weight in zip(dependency_type_group, old_group_weights):
                weights[t] = old_weight
            proposed_group_weights.append((result, new_weights))
        best_objective, best_weights = min(proposed_group_weights)
        toc = time.clock()
        if VERBOSE:
            print("best weights for %s are %s with objective %s"
                  % (dependency_type_group, best_weights, best_objective),
                  file=sys.stderr)
            print("time=%s" % (toc - tic), file=sys.stderr)
        if best_objective <= old_objective:
            for t, w in zip(dependency_type_group, best_weights):
                weights[t] = w
            old_objective = best_objective
        else:
            if VERBOSE:
                print("rejected new weights!", file=sys.stderr)
    return best_objective



def get_linearization(sentence, weights, thing_fn=DEFAULT_THING_FN):
    """ Get the deterministic linearization of a sentence given deptype weights.
    """
    def linearize_children(node):
        phrase_with_weights = [(0, node)]
        append_to = phrase_with_weights.append
        for h, d, dt in sentence.out_edges_iter(node, data=True):
            append_to((weights[thing_fn(sentence, (h, d, dt))], d))
        phrase_with_weights.sort()
        return [linearize_children(child_id) if child_id != node else node
                for _, child_id in phrase_with_weights]

    if 0 in sentence.node:
        return mindep.flatten(linearize_children(0))
    else:
        raise ValueError("Sentence did not have root node 0.")
    
def intervals(numbers, low, high):
    """ Get all intervals between the given numbers and the given low and
    high numbers.

    Numbers must be sorted.

    Example:
    >> list(intervals([-.5, 0, .25, .5], -1, 1))
    [(-1, -0.5), (-0.5, 0), (0, 0.25), (0.25, 0.5), (0.5, 1)]

    """
    for number in numbers:
        yield low, number
        low = number
    yield low, high

def get_interaction_table(corpus, thing_fn=DEFAULT_THING_FN):
    """ Return a dict of relation types t to the set of other relation types 
    that co-occur with t as direct children of some head."""
    d = defaultdict(set)
    for sentence in corpus.sentences():
        for node in sentence.nodes():
            deptypes = set(thing_fn(sentence, edge)
                           for edge in sentence.out_edges_iter(node, data=True))
            for deptype in deptypes:
                d[deptype] |= deptypes
    for deptype, d_set in d.items():
        d_set.remove(deptype)
    return d
            
def main(lang='en', thing_fn=None, objective_fn=None, num_initial_restarts=100):
    import time
    from corpora import corpora
    corpus = corpora[lang]
    corpus.load_into_memory()
    if VERBOSE:
        print("Loaded corpus into memory.", file=sys.stderr)
        print("%s sentences." % len(corpus.sentences()), file=sys.stderr)

    if thing_fn is None:
        thing_fn = DEFAULT_THING_FN
    else:
        thing_fn = eval(thing_fn)

    if objective_fn is None:
        objective_fn = test_set_surprisal
    else:
        objective_fn = eval(objective_fn)

    num_initial_restarts = int(num_initial_restarts)

    score, weights = find_optimal_weights(
        corpus,
        thing_fn=thing_fn,
        objective_fn=objective_fn,
        num_initial_restarts=num_initial_restarts,
    )
    print(score)
    for deptype, weight in weights.items():
        print("%s\t%s" % (deptype, weight))


if __name__ == '__main__':
    main(*sys.argv[1:])
    

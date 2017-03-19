#!/usr/bin/python3
from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import
from .compat import *

import random
import itertools

import networkx as nx

from . import depgraph

def flatten(iterable, _isinstance=isinstance):
    lst = []
    lst_extend = lst.extend
    lst_append = lst.append
    for item in iterable:
        if _isinstance(item, list):
            lst_extend(flatten(item))
        else:
            lst_append(item)
    return lst

def deplen_by_deptype(thing_fn, sentence, linearization=None, fn=sum):
    if linearization is None:
        linearization = sorted(sentence.nodes())
    deplens = Counter()
    for h_id, d_id, dt in sentence.edges_iter(data=True):
        if h_id != 0:
            try:
                linearization_index = {w_id : i
                                       for i, w_id in enumerate(linearization)}
                deplens[thing_fn(sentence, (h_id, d_id, dt))] += abs(linearization_index[h_id]
                                                                     - linearization_index[d_id])
            except ValueError:
                linearization_set = set(linearization)
                bad_nodes = [node for node in sentence.nodes()
                             if node not in linearization_set]
                error_str = "Could not get deplen for sentence because it "
                error_str += "contains nodes which aren't in the linearization:"
                error_str += bad_nodes
                raise ValueError(error_str)
    return deplens

def sum_of_abs(xys):
    return sum(abs(x - y) for x, y in xys)

def embedding_depths(sentence, linearization=None, include_root=False):
    if linearization is None:
        li = {w_id : w_id for w_id in sentence.nodes()}
    else:
        li = {w_id:i for i, w_id in enumerate(linearization)}

    try:
        for n in sentence.nodes():
            if n != 0 or include_root:
                # How many arcs are there over this node?
                num_arcs = 0
                for h, d in sentence.edges():
                    if h != 0 or include_root:
                        if (li[h] < li[n] < li[d]) or (li[d] < li[n] < li[h]):
                            num_arcs += 1
                yield num_arcs
    except KeyError:
        error_str = "Could not get embedding depths for sentence because it "
        error_str += "contains an edge "
        error_str += "which is not represented in the linearization: "
        error_str += str(linearization)
        error_str += "\nEdges: %s" % sentence.edges()
        error_str += "\nNodes: %s" % sentence.nodes()
        raise ValueError(error_str)            

def test_embedding_depths():
    s = nx.DiGraph([(0, 4), (1, 3), (3, 2), (4, 1), (4, 6), (6, 5)])
    depths = list(embedding_depths(s))
    assert depths == [0, 2, 1, 0, 1, 0]

    depths = list(embedding_depths(s, linearization=[0, 5, 6, 1, 2, 3, 4]))
    assert depths == [1, 3, 2, 0, 0, 0]

    s.add_edge(4, 7)
    depths = list(embedding_depths(s))
    assert depths == [0, 2, 1, 0, 2, 1, 0]

def sum_embedding_depth(*args, **kwds):
    return sum(embedding_depths(*args, **kwds))

def max_embedding_depth(*args, **kwds):
    return max(embedding_depths(*args, **kwds))

def deplen(sentence, linearization=None, include_root=False, fn=sum_of_abs, filters=None):
    if filters is None:
        filters = []
    if not include_root:
        filters.insert(0, lambda s, l, hd: hd[0] != 0)
    if linearization is None:
        linearization_index = {w_id : w_id for w_id in sentence.nodes()}
        return fn([(h_id, d_id)
                  for h_id, d_id in sentence.edges_iter()
                  if all(f(sentence, linearization_index, (h_id, d_id))
                         for f in filters)
                  ])
    else:
        try:
            linearization_index = {w_id:i for i, w_id in enumerate(linearization)}
            return fn([(linearization_index[h_id], linearization_index[d_id])
                    for h_id, d_id in sentence.edges_iter()
                    if all(f(sentence, linearization_index, (h_id, d_id))
                           for f in filters)
                       ])
        except KeyError:
            error_str = "Could not get deplen for sentence because it "
            error_str += "contains an edge "
            error_str += "which is not represented in the linearization: "
            error_str += str(linearization)
            error_str += "\nEdges: %s" % sentence.edges()
            error_str += "\nNodes: %s" % sentence.nodes()
            raise ValueError(error_str)

def filter_oracular_reductions(sentence, lin, hd):
    """in a structure with right-sisters A->B A->C A->D,
    a parser with an eager reduction rule can process A->B, A->C, and A->D 
    without incurring memory cost beyond A->D, because the memory stack
    grows as O(1) (Abney & Johnson, 1990).
    however, in a structure with left-sisters A<-D, B<-D, C<-D,
    there is no way for the memory stack to grow slower than O(n),
    so the correct cost is the sum of dependency lengths.
    this function filters out dependencies that should be ignored
    according to this kind of parsing oracle:
    return false for right dependencies h->d1 where there is some
    other dependency h->d2 where d1 < d2; otherwise true.
    """
    h, d = hd
    lin_d = lin[d]
    return (
        lin_d < lin[h]
        or all(lin[d_] <= lin_d for _,d_ in sentence.out_edges(h))
    )

def best_case_memory_cost(sentence,
                          linearization=None,
                          include_root=False,
                          fn=sum_of_abs,
                          filters=None):
    """ Memory cost for an aggressively reducing parsing oracle. """
    if filters is None:
        filters = [filter_oracular_reductions]
    else:
        filters = list(filters) # make sure it's not a tuple or whatever
        filters.append(filter_oracular_reductions)
    return deplen(sentence,
                  linearization=linearization,
                  include_root=include_root,
                  fn=fn,
                  filters=filters)

def worst_case_memory_cost(sentence,
                           linearization=None,
                           include_root=False,
                           fn=sum_of_abs,
                           filters=None):
    """ Worst-case memory cost for a projective parser. """
    TODO

def _randlin_projective(sentence,
                        head_final_bias=0,
                        move_head=True,
                        move_deps=True):
    def expand_randomly(word_id, linearization):
        word_position = linearization.index(word_id)
        children = [w_id for _, w_id in sentence.out_edges_iter(word_id)]
        if head_final_bias and random.random() < head_final_bias: # if flip, hf
            random.shuffle(children)
            children.append(word_id)
        elif not move_head: # otherwise, if the head is not mobile, keep it
            left_children = [n for n in children if n < word_id]
            right_children = [n for n in children if n > word_id]
            random.shuffle(left_children)
            random.shuffle(right_children)
            children = left_children
            children.append(word_id)
            children.extend(right_children)
        elif not move_deps: # otherwise, if deps aren't mobile, just move head
            i = random.choice(range(len(children)))
            children.insert(i, word_id)
        else: # otherwise, if the head is mobile, move it
            children.append(word_id)
            random.shuffle(children)
        linearization[word_position] = children
        for child_id in children:
            if child_id != word_id:
                expand_randomly(child_id, children)
    linearization = list(depgraph.roots_of(sentence))
    expand_randomly(0, linearization)
    return linearization

def randlin_projective(sentence,
                       head_final_bias=0,
                       move_head=True,
                       move_deps=True):
    assert move_head or move_deps
    linearization = _randlin_projective(sentence,
                                        head_final_bias=head_final_bias,
                                        move_head=move_head)
    flat_linearization = flatten(linearization)
    assert set(flat_linearization) == set(sentence.nodes())
    return deplen(sentence, flat_linearization), flat_linearization

def randlin_unconstrained(sentence):
    nodes = sentence.nodes()
    random.shuffle(nodes)
    return deplen(sentence, nodes), nodes

def mean_lin_samples(fn, sentence, num_samples=10):
    samples = (fn(sentence) for _ in range(num_samples))
    return mean(cost for cost, _ in samples)

def mindep_chung(sentence):
    """ Find the general minimal dependency length linearization of a tree.
    Algorithm 2 from Chung (1984)

    @article{chung1984optimal,
      author={F. R. K. Chung},
      year={1984},
      title={On optimal linear arrangements of trees},
      journal={Computers \& Mathematics with Applications},
      volume={10},
      number={1},
      page={43--60},
    }
    """
    # We assume that the tree is rooted.
    # From Chung:

    # g_n(T^*) := f_n(T^*) + \pi(r) - 1
    # g is the minimum of g_n
    # \pi is the map from nodes to integers

    # OLA \pi is of type (: T_0) or type (T_2, ..., T_{2p} : T_{2p+1}, ..., T_1)
    # by type (T_{i_t}, ..., T_{i_s} : T_{i_{s + 1}}, ..., T_{i_t}) we mean the
    # set of linear arrangements in which
    # V(T_{i_t}, ..., V(T_{i_s}), V(T - \Union_{k=1}^t T_{i_k})),
    # V(T_i{s+1}), ..., V(T_{i_t}) are labeled by consecutive integers in this
    # order.
    TODO

mindep = mindep_chung

def mindep_dp(sentence, head_boundary, boundary, generate_linearizations):
    """ General dynamic programming solution to the minimum dependency length
    linearization problem. """

    def reverse_if_needed(linearization,
                          containing_linearization,
                          word_id,
                          phrase_position):
        if word_id == 0:
            return
        word_position = linearization.index(word_id)
        head_id = depgraph.head_of(sentence, word_id)
        if head_id == 0:
            return
        head_position = containing_linearization.index(head_id)
        sizes = [
            depgraph.num_words_in_phrase(sentence, w_id)
            for w_id in linearization
        ]
        head_left = head_position < phrase_position
        head_right = not head_left
        left_size = sum(sizes[:word_position])
        right_size = sum(sizes[(word_position+1):])
        if ((head_left and left_size > right_size)
            or (head_right and right_size > left_size)):
            linearization.reverse()

    def reconstitute_linearization(result):
        def expand_word(word_id, linearization):
            if word_id in result:
                word_position = linearization.index(word_id)
                _, sublin = result[word_id]
                sublin = list(sublin)
                reverse_if_needed(sublin, linearization, word_id, word_position)
                linearization[word_position] = sublin
                for child_id in sublin:
                    if child_id != word_id:
                        expand_word(child_id, sublin)
            
        total_score = sum(score for score, _ in result.values())
        extended = set()
        linearization = [word_id for _, word_id in sentence.out_edges(0)]
        for word_id in linearization:
            expand_word(word_id, linearization)
        return total_score, flatten(linearization)
    
    def linearizations_and_costs(word_id):
        children = [w_id for _, w_id in sentence.out_edges(word_id)]
        linearizations = generate_linearizations(children, word_id)
        for linearization in linearizations:
            linearization = tuple(linearization) # for hashability
            d_to_head = head_boundary(linearization, word_id)
            d_to_deps = sum(boundary(linearization, word_id, w_id)
                            for w_id in children)
            cost = d_to_head + d_to_deps
            yield cost, linearization

    result = {word_id: min(linearizations_and_costs(word_id))
              for word_id in sentence.nodes() if word_id != 0}
    #return sum(cost for cost, _ in result.values())
    return reconstitute_linearization(result)

def mindep_projective_dp(sentence, generate_linearizations):
    def head_boundary(linearization, word_id):
        if depgraph.head_of(sentence, word_id) == 0:
            return 0
        word_position = linearization.index(word_id)
        sizes = [
            depgraph.num_words_in_phrase(sentence, w_id)
            for w_id in linearization
        ]
        return min(sum(sizes[:word_position]),
                   sum(sizes[(word_position+1):]))

    def boundary(linearization, head_id, dep_id):
        if head_id == 0:
            return 0
        head_position = linearization.index(head_id)
        dep_position = linearization.index(dep_id)
        sizes = [
            depgraph.num_words_in_phrase(sentence, w_id)
            for w_id in linearization
        ]
        if dep_position < head_position:
            return 1 + sum(sizes[(dep_position+1):head_position])
        else:
            return 1 + sum(sizes[(head_position+1):dep_position])

    return mindep_dp(sentence, head_boundary, boundary, generate_linearizations)

def mindep_projective_full(sentence):
    def generate_linearizations(words, head_id):
        return itertools.permutations(words + [head_id])
    return mindep_projective_dp(sentence, generate_linearizations)

def mindep_projective_full_head_final(sentence):
    def generate_linearizations(words, head_id):
        for perm in itertools.permutations(words):
            yield list(perm) + [head_id]
    return mindep_projective_dp(sentence, generate_linearizations)

def linearize_by_weight_head_final(sentence):
    def generate_linearizations(words, head_id):
        def weight(word_id):
            if word_id == head_id:
                return -1
            else:
                return depgraph.num_words_in_phrase(sentence, word_id)
        sorted_words = sorted(list(words) + [head_id], key=weight, reverse=True)
        return [sorted_words]
    return mindep_projective_dp(sentence, generate_linearizations)

def mindep_projective_alternating(sentence, move_head=True):
    def generate_linearizations(words, head_id):
        def weight(word_id):
            if word_id == head_id:
                return 0
            else:
                return depgraph.num_words_in_phrase(sentence, word_id)
        if move_head:
            sorted_words = sorted(list(words)+[head_id], key=weight)
            alternating = arrange_alternating_outward(sorted_words)
        else:
            left = [word for word in words if word < head_id]
            right = [word for word in words if word > head_id]
            left.sort(key=weight)
            left.reverse() # decreasing weight
            right.sort(key=weight) # increasing weight
            alternating = left
            alternating.append(head_id)
            alternating.extend(right)
        return [alternating, reversed(alternating)]
    return mindep_projective_dp(sentence, generate_linearizations)

def alternating_outward_indices():
    yield 0
    for i in itertools.count(1):
        yield -i
        yield i

def arrange_alternating_outward(xs):
    return [x for _, x in sorted(zip(alternating_outward_indices(), xs))]

def test_deplen():
    import corpora
    s = next(corpora.hamledt2_corpora['de'].sentences())
    assert deplen(s) == 16
    assert deplen(s, s.nodes()) == 16
    assert deplen(s, list(reversed(s.nodes()))) == 16

def test_best_case_memory_cost():
    import networkx as nx
    right_sisters = nx.DiGraph()
    right_sisters.add_edge(1, 2)
    right_sisters.add_edge(1, 3)
    right_sisters.add_edge(1, 4)
    assert best_case_memory_cost(right_sisters) == 3

    left_sisters = nx.DiGraph()
    left_sisters.add_edge(4, 1)
    left_sisters.add_edge(4, 2)
    left_sisters.add_edge(4, 3)
    assert best_case_memory_cost(left_sisters) == 6

    right_spine = nx.DiGraph()
    right_spine.add_edge(1, 2)
    right_spine.add_edge(2, 3)
    right_spine.add_edge(3, 4)
    assert best_case_memory_cost(right_spine) == 3

def test_randlin_projective():
    import corpora
    sentences = itertools.islice(corpora.hamledt2_corpora['ta'].sentences(), None, 1000)
    for s in sentences:
        random_deplen, random_linearization = randlin_projective(s)
        assert all(x in s.nodes() for x in random_linearization)
        assert all(x in random_linearization for x in s.nodes() if x != 0)

def test_mindep_projective_full():
    import corpora
    s = next(iter(corpora.hamledt2_corpora['de'].sentences()))
    score, lin = mindep_projective_full(s)
    assert score == 10
    assert deplen(s, lin) == 10

def test_mindep_projective_alternating():
    import corpora
    s = next(iter(corpora.hamledt2_corpora['de'].sentences()))
    score, lin = mindep_projective_alternating(s)
    assert score == 10
    assert deplen(s, lin) == 10

def test_alternating_optimal():
    from corpora import corpora    
    # verify that only looking at the alternating solution gives equivalent
    # results to DP for projective linearizations (Gildea & Temperley, 2007)
    sentences = itertools.islice(corpora['ta'].sentences(), None, 10)
    for s in sentences:
        scoref, linf = mindep_projective_full(s)
        scorea, lina = mindep_projective_alternating(s)
        assert scoref == scorea
        assert deplen(s, linf) == deplen(s, lina)


if __name__ == '__main__':
    import nose
    nose.runmodule()

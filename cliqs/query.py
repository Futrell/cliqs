from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from .compat import *

import operator
import itertools

import networkx as nx

from . import depgraph

_SENTINEL = object()

flat = itertools.chain.from_iterable

def matching_full_subtrees(query, target):
    """ Return subgraphs of target which match the query exactly, with no 
    further structure in the relevant nodes of target. """
    return matching_subtrees(operator.eq, query, target)

def matching_partial_subtrees(query, target):
    """ Return subgraphs of target which are isomorphic to query with arbitrary
    additional structure in the relevant nodes of target. """
    return matching_subtrees(operator.le, query, target)

def matching_strict_subtrees(query, target):
    """ Return subgraphs of target which are isomorphic to query without 
    additional structure among the relevant nodes of target, but possibly with
    additional edges emanating from the fringe nodes. """
    def f(q, t):
        return q == 0 or q == t
    return matching_subtrees(f, query, target)

def matching_subtrees(criterion, query, target):
    """ Yield all subgraphs of target matching the query. """
    for node in target.nodes():
        yield from match_subtree_at(criterion, query, target, node)

def test_matching_full_subtrees():
    t = nx.DiGraph([(0, 1), (0, 2), (1, 3), (2, 4), (2, 5)])
    q = nx.DiGraph([(0, 1), (0, 2)])
    full_matches = list(matching_full_subtrees(q, t))
    assert len(full_matches) == 1
    assert full_matches[0] == (2, 4, 5)

def test_matching_strict_subtrees():
    t = nx.DiGraph([(0, 1), (0, 2), (1, 3), (2, 4), (2, 5)])
    q = nx.DiGraph([(0, 1), (0, 2)])    
    strict_matches = set(matching_strict_subtrees(q, t))
    assert strict_matches == {(0, 1, 2), (2, 4, 5)}

    t = nx.DiGraph([(1, 2), (1, 3), (1, 4)])
    q = nx.DiGraph([(0, 1), (0, 2)])
    strict_matches = set(matching_strict_subtrees(q, t))
    assert strict_matches == set()

def test_matching_partial_subtrees():    
    t = nx.DiGraph([(0, 1), (0, 2), (2, 3), (2, 4), (4, 5)])
    q = nx.DiGraph([(0, 1), (0, 2), (2, 3)])
    partial_matches = set(matching_partial_subtrees(q, t))
    assert partial_matches == {(0, 1, 2, 3), (0, 1, 2, 4), (2, 3, 4, 5)}

def match_subtree_at(criterion, query, target, node):
    """ Yield the bunches of the nodes of the target graph that match the query 
    graph at the given node. For result r, nx.subgraph(target, r) is isomorphic 
    to query and the annotations of query are a subset of the annotations of 
    target.
    """
    root = depgraph.root_of(query)
    return _match_subtree_at(criterion, query, root, target, node)

def field_matches(q, t_value):
    if hasattr(q, '__call__'):
        return q(t_value)
    else:
        return q == t_value

def _match_subtree_at(criterion, q, q_node, t, t_node):
    q_children = depgraph.left_right_dependents_of(q, q_node)
    all_t_children = depgraph.left_right_dependents_of(t, t_node)
    t_children_subsets = itertools.product(
        itertools.combinations(all_t_children.left, len(q_children.left)),
        itertools.combinations(all_t_children.right, len(q_children.right))
    )
    
    def node_conditions():
        # Structure is the same from this node out
        yield (
            criterion(len(q_children.left), len(all_t_children.left))
            and criterion(len(q_children.right), len(all_t_children.right))
        )

        # All query node attributes present in target tree
        for a, q_value in q.node[q_node].items():
            yield field_matches(q_value, t.node[t_node].get(a, _SENTINEL))

    def child_conditions(t_children):
        # All template edge attributes present in target tree
        # Assumes the q_children map to the t_children in a zip.
        for i in [0, 1]:
            for q_child, t_child in zip(q_children[i], t_children[i]):
                for a, q_value in q.edge[q_node][q_child].items():
                    yield field_matches(
                        q_value,
                        t.edge[t_node][t_child].get(a, _SENTINEL)
                    )

    def rest(t_children): 
        # This generator checks the conditions recursively down the tree,
        # for given children. It yields the matching node bunches.
        def subtrees():
            for i in [0, 1]:
                for q_child, t_child in zip(q_children[i], t_children[i]):
                    yield _match_subtree_at(criterion, q, q_child, t, t_child)

        # subtrees() yields up sets of nodes for each child;
        # each set is a possible match rooted in that child.
        # For example,
        # subtrees() might give [[[1, 2], [5, 6]], [[3]], [[4]]].
        # This means the first child matches at either [1, 2] or [5, 6],
        # the second child matches at only [3], and the third matches
        # at only [4].
        # Now we want to yield [1, 2, 3, 4] and [5, 6, 3, 4].
        # Those are unions of the nodes in possible matches.
        # So we first do a product over *subtrees(), which gives
        # [[[1, 2], [3], [4]], [[5, 6], [3], [4]]].
        # This causes the node sets in subtrees to be evaluated,
        # but the product over those sets is a lazy iterator.
        results = itertools.product(*subtrees())
        
        # these are the two possible overall matches, structured internally
        # based on which children they are rooted in. Then we want to flatten
        # that structure so we map flat over each result.
        return map(flat, results)
            
    if all(node_conditions()):
        for t_children in t_children_subsets:
            if all(child_conditions(t_children)):
                for subtree in rest(t_children):
                    yield (t_node,) + tuple(subtree)
                    
def survey(sentences):
    interesting_templates = {
        'rightward_chain': nx.DiGraph([(0, 1), (1, 2)]),
        'leftward_chain': nx.DiGraph([(2, 1), (1, 0)]),
        'rightward_flat': nx.DiGraph([(0, 1), (0, 2)]),
        'leftward_flat': nx.DiGraph([(2, 0), (2, 1)]),
        'rightward_inner': nx.DiGraph([(0, 2), (2, 1)]),
        'leftward_inner': nx.DiGraph([(2, 0), (0, 1)]),
    }

    return {
        name : matching_strict_subtrees(t, sentences)
        for name, t in interesting_templates
    }

# TODO make matches sensitive to the position of the head,
# not just the order of the dependents

def embedding_survey(sentences):
    queries = {
        'rc': nx.DiGraph([(0, 1)]),
        'orc': nx.DiGraph([(0, 3), (3, 1), (3, 2)]),
        'subject_rc': nx.DiGraph([(2, 0), (0, 1)]),
        'rc_containing_preverbal_rc': nx.DiGraph([(0, 3), (3, 1), (1, 2)]),
        'rc_containing_postverbal_rc': nx.DiGraph([(0, 1), (3, 1), (1, 2)]),
        'pp': nx.DiGraph([(0, 2), (2, 1)]),
    }
    queries['intransitive_verb'].node[1]['pos'] = 'VERB'
    queries['intransitive_verb'].edge[1][0]['deptype'] = 'nsubj'

    queries['v_medial_transitive_verb'].node[1]['pos'] = 'VERB'
    queries['v_medial_transitive_verb'].edge[1][0]['deptype'] = 'nsubj'
    queries['v_medial_transitive_verb'].edge[1][2]['deptype'] = 'dobj'

    queries['v_final_transitive_verb'].node[2]['pos'] = 'VERB'
    queries['v_final_transitive_verb'].edge[2][0]['deptype'] = 'nsubj'
    queries['v_final_transitive_verb'].edge[2][1]['deptype'] = 'dobj'

    queries['rc'].edge[0][1]['deptype'] = lambda x: x.startswith('acl')
    queries['rc'].node[0]['pos'] = 'NOUN'
    queries['rc'].node[1]['pos'] = 'VERB' 

    queries['orc'].node[0]['pos'] = 'NOUN'
    queries['orc'].node[1]['pos'] = 'PRON'
    queries['orc'].node[2]['pos'] = 'NOUN'
    queries['orc'].node[3]['pos'] = 'VERB'    
    queries['orc'].edge[0][3]['deptype'] = lambda x: x.startswith('acl')
    queries['orc'].edge[3][2]['deptype'] = 'nsubj'
    queries['orc'].edge[3][1]['deptype'] = 'dobj'

    queries['subject_rc'].node[0]['pos'] = 'NOUN'
    queries['subject_rc'].node[1]['pos'] = 'VERB'
    queries['subject_rc'].node[2]['pos'] = 'VERB'
    queries['subject_rc'].edge[2][0]['deptype'] = 'nsubj'
    queries['subject_rc'].edge[0][1]['deptype'] = lambda x: x.startswith('acl')

    def gen(q):
        for s in sentences:
            for match in matching_partial_subtrees(q, s):
                yield (s, match)
            
    return {
        name : list(gen(q))
        for name, q in queries.items()
    }

def simple_survey(sentences):
    queries = {}
    queries['hd0'] = nx.DiGraph([(0, 1)])
    queries['hd1'] = nx.DiGraph([(1, 0)])
    
    queries['gd0'] = nx.DiGraph([(0, 1), (1, 2)])
    queries['gd1'] = nx.DiGraph([(0, 2), (2, 1)])
    queries['gd2'] = nx.DiGraph([(1, 0), (0, 2)])
    queries['gd3'] = nx.DiGraph([(1, 2), (2, 0)])
    queries['gd4'] = nx.DiGraph([(2, 0), (0, 1)])
    queries['gd5'] = nx.DiGraph([(2, 1), (1, 0)])

    queries['ss0'] = nx.DiGraph([(0, 1), (0, 2)])
    queries['ss1'] = nx.DiGraph([(1, 0), (1, 2)])
    queries['ss2'] = nx.DiGraph([(2, 0), (2, 1)])
    
    def gen(q):
        for s in sentences:
            for match in matching_partial_subtrees(q, s):
                yield (s, match)

    result = {
        name : list(gen(q))
        for name, q in queries.items()
    }

    d = {}
    d['hd'] = list(flat(result['hd'+k] for k in "01"))
    d['gd'] = list(flat(result['gd'+k] for k in "012345"))
    d['ss'] = list(flat(result['ss'+k] for k in "012"))

    return d

if __name__ == '__main__':
    import nose
    nose.runmodule()

""" deptransform
Functions of type dependency graph -> dependency graph

There is no guarantee that these functions leave the input graph unmutated;
to ensure purity, call a function f(sentence, *a) as immutably(f)(sentence, *a).
"""
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import
from .compat import *

import sys
import copy
import functools

import networkx as nx
import rfutils

from .depgraph import heads_of, head_of, dependents_of, gap_degree, words_of, root_of, roots_of

VERBOSE = False
STRICT = True # eliminate sentences that cannot be transformed from CH
STRICT_COLLAPSE = False # eliminate sentences where flats cannot be collapsed
STRICT_PROJECTIVITY = False # eliminate sentences where CH conversion creates nonprojectivity

def immutably(f):
    @functools.wraps(f)
    def wrapped(sentence, *a, **k):
        sentence_ = copy.deepcopy(sentence)
        return f(sentence_, *a, **k)
    return wrapped

def sequence_continuous(xs):
    xs = list(xs)
    sliding = zip(xs, xs[1:])
    return all(x+1 == y for x, y in sliding)

FLAT_RELS = frozenset({
    'flat', # TODO make sure this should be / instead of :
    'flat:name',
    'flat:foreign',
    'flat:repeat',
    'flat:title',
    'goeswith',
    'fixed'
})
def collapse_flat(sentence, rels=FLAT_RELS, verbose=VERBOSE):
    original_sentence = copy.deepcopy(sentence)
    original_words = " ".join(map(str, words_of(sentence)))
    nodes = list(reversed(sorted(sentence.nodes())))
    nodes_hitlist = []
    edges_hitlist = []
    root = root_of(sentence)
    nodes = list(reversed([d for _, d in nx.bfs_edges(sentence, root)]))
    nodes.append(root)
    for h in nodes:
        flats = sorted([
            d for _, d, dt in sentence.out_edges(h, data=True)
            if dt['deptype'] in rels
        ])
        if flats and sequence_continuous(flats):
            words = sorted(flats + [h])
            new_word = " ".join(sentence.node[d]['word'] for d in words)
            sentence.node[h]['word'] = new_word
            nodes_hitlist.extend(flats)
            edges_hitlist.extend((h,d) for d in flats)
            for d in flats:
                for _, dd, dt in sentence.out_edges(d, data=True):
                    if dt['deptype'] not in rels:
                        sentence.add_edge(h, dd, dt)
        elif not sequence_continuous(flats) and verbose:
            print("Discontinuous flat at line %s node %s" % (
                sentence.start_line, str(h)
            ), file=sys.stderr)
    for h, d in edges_hitlist:
        sentence.remove_edge(h, d)
    for node in nodes_hitlist:
        sentence.remove_node(node)
    new_words = " ".join(map(str, words_of(sentence)))
    # Make sure nothing went wrong. If it did, ditch this process.
    if original_words != new_words or len(list(roots_of(sentence))) > 1:
        if verbose:
            print("Warning, could not collapse sentence start line %s" % sentence.start_line,
                    file=sys.stderr)
        if STRICT_COLLAPSE:
            return None
        else:
            return original_sentence
    return renumber_words(sentence)

def test_collapse_flat():
    # most basic case
    s = nx.DiGraph([(0, 1), (0, 2), (0, 3), (0, 4)])
    for i, letter in enumerate("abcde"):
        s.node[i]['word'] = letter
    s.edge[0][1]['deptype'] = 'flat'
    s.edge[0][2]['deptype'] = 'flat'
    s.edge[0][3]['deptype'] = 'flat'
    s.edge[0][4]['deptype'] = 'flat'

    s2 = immutably(collapse_flat)(s, rels={'flat'})
    assert len(s2) == 1
    assert s2.node[0]['word'] == "a b c d e"


    # this scenario should not arise in UD 2.1; but it does.
    # see af/all.conllu line 4867
    s = nx.DiGraph([(0, 1), (1, 2), (2, 3)])
    for i, letter in enumerate("abcd"):
        s.node[i]['word'] = letter
    s.edge[0][1]['deptype'] = 'flat'
    s.edge[1][2]['deptype'] = 'flat'
    s.edge[2][3]['deptype'] = 'flat'

    s = nx.DiGraph([(3, 2), (2, 1), (1, 0)])
    for i, letter in enumerate("abcd"):
        s.node[i]['word'] = letter
    s.edge[3][2]['deptype'] = 'flat'
    s.edge[2][1]['deptype'] = 'flat'
    s.edge[1][0]['deptype'] = 'flat'
    
    s2 = immutably(collapse_flat)(s, rels={'flat'})
    # also bad, but occurs in af/all.conllu line 18801
    assert len(s2) == 1
    assert s2.node[0]['word'] == "a b c d"

    s = nx.DiGraph([(3, 0), (3, 1), (3, 2)])
    for i, letter in enumerate("abcd"):
        s.node[i]['word'] = letter
    s.edge[3][0]['deptype'] = 'flat'
    s.edge[3][1]['deptype'] = 'flat'
    s.edge[3][2]['deptype'] = 'flat'
    
    s2 = immutably(collapse_flat)(s, rels={'flat'})
    # might as well try this possibility
    assert len(s2) == 1
    assert s2.node[0]['word'] == "a b c d"

    s = nx.DiGraph([(0, 1), (0, 2), (0, 3), (3, 4)])
    for i, letter in enumerate("abcde"):
        s.node[i]['word'] = letter
    s.edge[0][1]['deptype'] = 'flat'
    s.edge[0][2]['deptype'] = 'flat'
    s.edge[0][3]['deptype'] = 'flat'
    s.edge[3][4]['deptype'] = 'other'

    # shouldn't happen in UD, but nonetheless it does
    s2 = immutably(collapse_flat)(s, rels={'flat'})
    assert len(s2) == 2
    assert s2.node[0]['word'] == "a b c d"
    assert s2.edge[0][1]['deptype'] == 'other'

# remove everything except nouns, verbs, adjectives, adverbs, numerals,
FW_POS = frozenset("ADP AUX CCONJ DET PART PRON SCONJ PUNCT".split())
FW_RELS = frozenset("aux case cc det expl mark punct")
def remove_function_words(sentence, verbose=VERBOSE):
    return remove_from_sentence(
        sentence,
        badpos=FW_POS,
        badrel=FW_RELS,
        verbose=VERBOSE,
        strict=False,
    )
    
PUNCTUATION_POS = frozenset(". punc punct PUNC PUNCT wp".split())
PUNCTUATION_RELS = frozenset("punct p WP".split())
def remove_punct_from_sentence(sentence, verbose=VERBOSE):
    return remove_from_sentence(
        sentence,
        badpos=PUNCTUATION_POS,
        badrel=PUNCTUATION_RELS,
        verbose=VERBOSE,
        strict=True,
    )

def remove_from_sentence(sentence, badrel, badpos, verbose=VERBOSE, strict=False): 
    """ Destructively remove punctuation from the given sentence. """
    edges_to_die = [
        (h,d,t) for h,d,t in sentence.edges(data='deptype')
        if t.split(":")[0] in badrel # TODO make sure it's : not /
    ]
    words_to_die = {d for _, d, _ in edges_to_die}
    words_to_die |= {
        w for w, w_attr in sentence.nodes(data=True)
        if w_attr.get('pos') in badpos
    }
    punctuation_as_head = {
        (h, d, t)
        for h,d,t in sentence.edges(data='deptype')
        if h in words_to_die
    }
    more_edges_to_die = [
        (h,d) for h,d in sentence.edges()
        if h in words_to_die or d in words_to_die
    ]

    for punct, word2, deptype in punctuation_as_head:
        # word1 -a-> punct -b-> word2
        # We want to remove punct and the relation -a->, and connect word1 to word2 with -b->
        # In the case of word -> punct -> punct -> word, remove all the intervening puncts.
        # Do not do this if word1 is root.
        if sentence.node[punct].get('pos') not in badrel:
            if verbose and strict:
                print(
                    "Head with illegal relation type! {}-{}->, {}".format(
                        sentence.node[punct].get('pos'),
                        deptype,
                        str(sentence.start_line)
                    ),
                    file=sys.stderr
                )
            if strict:
                return None
        heads = heads_of(sentence, punct)
        if len(heads) > 1:
            if verbose:
                print(
                    "Illegal word with multiple heads! {}".format(str(sentence.start_line)),
                    file=sys.stderr
                )
            return None
        word1 = heads[0]
        if word1 == 0:
            if verbose:
                print(
                    "Sentence with illegal root! {}".format(str(sentence.start_line)),
                    file=sys.stderr
                )
            return None
        else:
            while (sentence.node[word1].get('pos') in badpos
                              or deptype.split(":")[0] in badrel):
                head_of_word1 = head_of(sentence, word1)
                deptype = sentence.edge[head_of_word1][word1].get('deptype')
                word1 = head_of_word1
        sentence.add_edge(word1, word2, deptype=deptype)
        sentence.remove_edge(punct, word2)

    sentence.remove_edges_from(edges_to_die)
    sentence.remove_nodes_from(words_to_die)    
    sentence.remove_edges_from(more_edges_to_die)

    if not sentence.edges():
        if verbose:
            print("Empty sentence! {}".format(str(sentence.start_line)), file=sys.stderr)
        return None

    assert len(list(roots_of(sentence))) == 1

    return renumber_words(sentence)

def test_remove_punct_from_sentence():
    s = nx.DiGraph()
    s.add_edges_from([
        (0, 3, {'deptype': 'root'}),
        (3, 1, {'deptype': 'nsubj'}),
        (3, 2, {'deptype': 'punct'}),
        (3, 4, {'deptype': 'dobj'}),
        (3, 5, {'deptype': 'punct'}),
        (4, 6, {'deptype': 'punct'}),
        (6, 7, {'deptype': 'hello'}),
    ])
    s.node[1]['pos'] = 'NN'
    s.node[2]['pos'] = 'PUNCT'
    s.node[3]['pos'] = 'VV'
    s.node[4]['pos'] = 'PRP'
    s.node[5]['pos'] = 'PUNCT'
    s.node[6]['pos'] = 'PUNCT'
    s.node[7]['pos'] = 'HELLO'
    s.whatever = 'test'

    s2 = immutably(remove_punct_from_sentence)(s)
    assert set(s2.edges(data='deptype')) == {
        (0, 2, 'root'),
        (2, 1, 'nsubj'),
        (2, 3, 'dobj'),
        (3, 4, 'hello'),
    }
    assert s2.node[1]['pos'] == 'NN'
    assert s2.node[2]['pos'] == 'VV'
    assert s2.node[3]['pos'] == 'PRP'
    assert s2.node[4]['pos'] == 'HELLO'
    assert s2.whatever == 'test' # all attributes maintained
                      
def renumber_words(sentence):
    """ Destructively renumber the nodes of a graph G to range(0, len(G)) """
    old_nodes = sorted(sentence.nodes())
    new_node_mapping = {j:i for i,j in enumerate(old_nodes)}
    new_sentence = nx.relabel_nodes(sentence, new_node_mapping, copy=False)
    for w_id, word in new_sentence.nodes(data=True):
        word['id'] = w_id
    return new_sentence

def test_reverse_content_head_simple():
    # "on Sunday I was man that died"
    #   1      2 3   4   5    6    7
    # In Universal Dependencies 1.0 standard:
    s = nx.DiGraph([
        (0, 5, {'deptype': 'root'}), # 0 -root-> man
        (5, 2, {'deptype': 'nmod'}), # man -nmod-> Sunday
        (2, 1, {'deptype': 'case'}), # Sunday -case-> on
        (5, 4, {'deptype': 'cop'}),  # man -cop-> was
        (5, 3, {'deptype': 'nsubj'}), # man -nsubj-> I
        (5, 7, {'deptype': 'relcl'}), # man -relcl-> died
        (7, 6, {'deptype': 'mark'}), # died -mark-> that
    ])
    s.node[1]['pos'] = 'ADP' # on
    s.node[2]['pos'] = 'NOUN'  # Sunday
    s.node[3]['pos'] = 'PRON' # I
    s.node[4]['pos'] = 'VERB' # was
    s.node[5]['pos'] = 'NOUN'  # man
    s.node[6]['pos'] = 'SCONJ' # that
    s.node[7]['pos'] = 'VERB' # died

    s2 = immutably(reverse_content_head)(s, 'case', high_rels={})
    assert set(s2.edges(data='deptype')) == {
        (0, 5, 'root'),
        (5, 1, 'nmod'),
        (1, 2, 'case'),
        (5, 4, 'cop'),
        (5, 3, 'nsubj'),
        (5, 7, 'relcl'),
        (7, 6, 'mark'),
    }

    s2 = immutably(reverse_content_head)(s, 'mark', high_rels={})
    assert set(s2.edges(data='deptype')) == {
        (0, 5, 'root'),
        (5, 2, 'nmod'),
        (2, 1, 'case'),
        (5, 4, 'cop'),
        (5, 3, 'nsubj'),
        (5, 6, 'relcl'),
        (6, 7, 'mark'),
    }

    s2 = immutably(reverse_content_head)(s, 'cop', high_rels={'nmod', 'nsubj'})
    assert set(s2.edges(data='deptype')) == {
        (0, 4, 'root'),
        (4, 3, 'nsubj'),
        (4, 2, 'nmod'),
        (2, 1, 'case'),
        (4, 5, 'cop'),
        (5, 7, 'relcl'),
        (7, 6, 'mark'),
    }

    # today 's incident
    #   1   2     3
    s = nx.DiGraph([
        (0, 3, {'deptype': 'root'}),
        (3, 1, {'deptype': 'nmod:poss'}),
        (1, 2, {'deptype': 'case'}),
    ])

    s2 = immutably(reverse_content_head)(s, 'case', {}, verbose=False)
    assert set(s2.edges(data='deptype')) == {
        (0, 3, 'root'),
        (3, 2, 'nmod:poss'),
        (2, 1, 'case'),
    }

def test_reverse_content_head_complex():    
    # that he was under suspicion
    #   1   2  3    4      5
    s = nx.DiGraph([
        (0, 5, {'deptype': 'root'}),
        (5, 1, {'deptype': 'mark'}), # suspicion -mark-> that
        (5, 2, {'deptype': 'nsubj'}), # suspicion -nsubj-> he
        (5, 3, {'deptype': 'cop'}), # suspicion -cop-> was
        (5, 4, {'deptype': 'case'}), # suspicion -case-> under
    ])
    # currently:
    # suspicion -mark-> that
    #           -nsubj-> he
    #           -cop-> was
    #           -case-> under
    
    # step 1: case reversal; mark, nsubj and cop attach high:
    # under -mark-> that
    #       -nsubj-> he
    #       -cop-> was
    #       -case-> suspicion
    s2 = immutably(reverse_content_head)(s, 'case', {'mark', 'nsubj', 'cop', 'case'})
    
    # step 2: cop reversal; mark and nsubj attach high
    # was -cop-> under -case-> suspicion
    #     -mark-> that
    #     -nsubj-> he
    s3 = immutably(reverse_content_head)(s2, 'cop', {'mark', 'nsubj', 'cop'})
    
    # step 3: mark reversal; nothing attaches high
    # that -> was -> he
    #             -> under -> suspicion
    s4 = immutably(reverse_content_head)(s3, 'mark', {})
    assert set(s4.edges(data='deptype')) == {
        (0, 1, 'root'),
        (1, 3, 'mark'),
        (3, 2, 'nsubj'),
        (3, 4, 'cop'),
        (4, 5, 'case'),
    }

def lift_head(sentence, n1, n2, high_rels):
    """ Given a sentence with dependencies of form h -a-> n1 -b-> n2, 
    lift n2 to be the head of n1, as in h -a-> n2 -b-> n1. 
    Dependents of n1 matching high_rels are attached as dependents to n2.
    Destructive.
    """
    for h in heads_of(sentence, n1):
        r1 = sentence.edge[h][n1]['deptype']
        sentence.remove_edge(h, n1)
        sentence.add_edge(h, n2, deptype=r1)
    r2 = sentence.edge[n1][n2]['deptype']
    sentence.remove_edge(n1, n2)
    sentence.add_edge(n2, n1, deptype=r2)
    # at this point, all old dependents of n1 are still attached to n1.
    # Now lift the things in high_rels:
    for d in dependents_of(sentence, n1):
        if attach_match(sentence, high_rels, n1, d):
            r = sentence.edge[n1][d]['deptype']
            sentence.remove_edge(n1, d)
            sentence.add_edge(n2, d, deptype=r)
    return sentence

def test_lift_head():
    # "barely in big house"
    #       1  2   3     4
    s = nx.DiGraph([
        (0, 4, {'deptype': 'nmod'}),
        (4, 1, {'deptype': 'advmod'}),
        (4, 2, {'deptype': 'case'}),
        (4, 3, {'deptype': 'amod'}),
    ])
    s2 = immutably(lift_head)(s, 4, 2, {'advmod'})
    assert set(s2.edges(data='deptype')) == {
        (0, 2, 'nmod'),
        (2, 4, 'case'),
        (2, 1, 'advmod'),
        (4, 3, 'amod'),
    }

def attach_match(sentence, rels, n1, n2):
    # this could be expanded as a general pattern matching DSL...
    # for now, just match on deptype
    rel = sentence.edge[n1][n2]['deptype']
    return rel in rels or rel.split(":")[0] in rels

def reversible_paths(sentence, rel, verbose=VERBOSE):
    untouchables = set()
    #edge = copy.deepcopy(sentence.edge)
    for h in sentence.nodes():
        #outs = edge[h] 
        #relevant_outs = [d for d, dt in outs.items() if dt['deptype'] == rel]
        outs = sentence.out_edges(h, data='deptype')
        relevant_outs = [d for _, d, r in outs if r == rel]
        if len(relevant_outs) == 0:
            pass
        elif len(relevant_outs) == 1:
            if h in untouchables:
                pass
            else:
                d = relevant_outs[0]
                if any(r == rel for _, _, r in sentence.out_edges(d, data='deptype')):
                    if verbose:
                        print("Unfixable content-head dependency (chain) from node %s in sentence %s type %s" % (h, sentence, rel), file=sys.stderr)
                    yield None
                else:
                    yield h, d
                    untouchables.add(d)
        else:
            if verbose:
                print("Unfixable content-head dependency (sisters) from node %s in sentence %s type %s" % (h, sentence, rel), file=sys.stderr)
            yield None

def test_reversible_paths():
    # today 's incident
    #   1   2     3
    s = nx.DiGraph([
        (0, 3, {'deptype': 'root'}),
        (3, 1, {'deptype': 'nmod:poss'}),
        (1, 2, {'deptype': 'case'}),
        (3, 4, {'deptype': 'something'}),
        (4, 5, {'deptype': 'case'}),
        (5, 6, {'deptype': 'case'}),
    ])    
    paths = set(reversible_paths(s, 'case', verbose=False))
    assert (1, 2) in paths
    assert None in paths
    assert (5, 6) in paths or (4, 5) in paths

def reverse_content_head(sentence,
                         rel,
                         high_rels,
                         verbose=VERBOSE,
                         strict=STRICT,
                         strict_projectivity=STRICT_PROJECTIVITY):
    """ Destructively convert dependencies of the form A -x-> B -y-> C, where y
    is of type in rels, to A -x-> C -y-> B.
    Thus, complementizers head their sentences and prepositions head their
    nouns, rather than vice versa.
    """
    if verbose or strict_projectivity:
        old_gap_degree = gap_degree(sentence)
    for path in reversible_paths(sentence, rel, verbose=verbose):
        if path is None:
            if strict:
                if verbose:
                    print("Giving up on sentence %s"%sentence, file=sys.stderr)
                return None
        else:
            h, d = path
            sentence = lift_head(sentence, h, d, high_rels)
    if (verbose or strict_projectivity) and gap_degree(sentence)>old_gap_degree:
        if verbose:
            print(
                "Created nonprojectivity in sentence %s" % sentence,
                file=sys.stderr
            )
        elif strict_projectivity:
            if verbose:
                print("Giving up on sentence %s"  % sentence, file=sys.stderr)
            return None
    return sentence

if __name__ == '__main__':
    import nose
    nose.runmodule()

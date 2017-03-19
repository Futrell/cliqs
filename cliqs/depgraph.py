#!/usr/bin/python3
""" Functions on dependency graphs represented as objects with the 
networkx.DiGraph interface:

Attributes:
node
edge

Methods:
nodes_iter(data=True)
edges_iter(data='something')
predecessors(n)
predecessors_iter(n)
successors_iter(n)

Amenable to nx.descendants and nx.has_path

"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
from __future__ import absolute_import
from .compat import *

import bisect
import os.path
import operator
import tempfile
import functools
from collections import Counter, namedtuple, defaultdict

import networkx as nx
from rfutils import sliding, the_only


class EquableDiGraph(nx.DiGraph):
    def __eq__(self, other):
        return self.node == other.node and self.edge == other.edge


# get_attr : String -> (DiGraph x Int -> Maybe String)
def get_attr(attr):
    """ Produce a function which gets the given attribute of a sentence at a
    node, returning None if the attribute cannot be found. """
    def get(s, n):
        return s.node[n].get(attr)
    return get


# attr_of : String x DiGraph -> [String]
def attr_of(attr, s):
    """ Reduce a sentence to an iterable of attribute values,
    one for each non-root node. """
    return [s.node[n].get(attr, None) for n in s.nodes()]


words_of = functools.partial(attr_of, 'word')
lemmas_of = functools.partial(attr_of, 'lemma')


# phrase_of : DiGraph x Int -> [Int]
def phrase_of(s, word_id):
    """ Return a node index and the indices of all its transitive descendents,
    in order. """
    words = sorted(nx.descendants(s, word_id))
    bisect.insort(words, word_id)
    return words


def test_phrase_of():
    t = nx.DiGraph([(3, 2), (2, 1), (2, 0), (3, 5), (5, 4), (5, 6)])
    assert phrase_of(t, 0) == [0]
    assert phrase_of(t, 2) == [0, 1, 2]
    assert phrase_of(t, 5) == [4, 5, 6]
    assert phrase_of(t, 3) == [0, 1, 2, 3, 4, 5, 6]


def draw_sentence(s, **kwds):
    import nxpd
    for node in s.nodes_iter():
        attr = s.node[node]
        if 'word' not in attr:
            attr['label'] = str(node)
        elif attr['word'].startswith('_'):
            attr['label'] = attr['pos'] + '_%s' % node
        else:
            attr['label'] = attr['word'] + '_%s' % node
        attr['label'] = attr['label'].replace(":", "/")
    for e1, e2 in s.edges_iter():
        attr = s.edge[e1][e2]
        if 'deptype' in attr:
            attr['label'] = attr['deptype']
        else:
            attr['label'] = 'NONE'
        attr['label'] = attr['label'].replace(":", "/")
    nxpd.draw(s, **kwds)


def sentence_to_latex(s, with_deplen=False):
    words = [
        latex_escape(node.get('word', str(n)))
        for n, node in s.nodes_iter(data=True)
    ]
    deptext = " \& ".join(words)

    def label(h, d, dt):
        if with_deplen:
            return abs(h - d)
        elif dt is None:
            return ''
        else:
            return dt
    depedges = "\n".join(
        "\depedge{%s}{%s}{%s}" % (h + 1, d + 1, label(h, d, dt))
        for h, d, dt in s.edges_iter(data='deptype')
        if dt != 'root'
    )
    return LATEX_DEPENDENCY_TEMPLATE % (deptext, depedges)


LATEX_DEPENDENCY_TEMPLATE = """
\\begin{dependency}[theme=simple]
  \\begin{deptext}
    %s \\\\
  \\end{deptext}
    %s
\\end{dependency}
"""

LATEX_WIDE_DOCUMENT_TEMPLATE = """
\\documentclass{article}
\\usepackage[landscape]{geometry}
\\usepackage{tikz-dependency}
\\usepackage[utf8]{inputenc}
\\begin{document}
%s
\\end{document}
"""

LATEX_DOCUMENT_TEMPLATE = """
\\documentclass{article}
\\usepackage{tikz-dependency}
\\usepackage[utf8]{inputenc}
\\usepackage[normalem]{ulem}
\\begin{document}
%s
\\end{document}
"""

TO_ESCAPE = frozenset("$")


def latex_escape(xs):
    def gen():
        for x in xs:
            if x in TO_ESCAPE:
                yield "\\"
            yield x
    return "".join(gen())


def to_latex_document(content):
    """ Insert content into a LaTeX document template. """
    return LATEX_WIDE_DOCUMENT_TEMPLATE % content


def show_latex(doctext, cleanup=False):
    """ Show a pdf of a LaTeX document with doctext. 
    Only really expected to work on OS X.
    """
    import sh
    with tempfile.NamedTemporaryFile(mode='w+t', encoding='utf-8') as docout:
        print(doctext, file=docout)
        docout.flush()
        sh.pdflatex(docout.name)
        output_filename = os.path.basename(docout.name) + ".pdf"
        sh.open(output_filename)
        if cleanup:
            sh.rm(output_filename)
            #try:
            #    sh.clean_tex()
            #except sh.ErrorReturnCode_1:
            #    pass


def show_sentence_latex(s, **kwds):
    return show_latex(to_latex_document(sentence_to_latex(s, **kwds)))


def show_sentences_latex(ss, **kwds):
    text = "\n".join(sentence_to_latex(s, **kwds) for s in ss)
    show_latex(to_latex_document(text))


# roots_of : DiGraph -> Iterator Int     
def roots_of(s):
    """ Yield the root nodes of a sentence. """
    for node, in_degree in s.in_degree().items():
        if in_degree == 0:
            yield node


def test_roots_of():
    t = nx.DiGraph([(0, 1), (1, 2), (2, 3)])
    assert list(roots_of(t)) == [0]

    g = nx.DiGraph([(0, 1), (2, 3)])
    assert sorted(roots_of(g)) == [0, 2]

    g = nx.DiGraph([(0, 1), (2, 1)])
    assert sorted(roots_of(g)) == [0, 2]


# root_of : DiGraph -> Int    
def root_of(s):
    """ Return the single root node of a sentence; 
    die if there are 0 roots or more than 1 root. 
    """
    return the_only(roots_of(s))


def test_root_of():
    t = nx.DiGraph([(2, 0), (0, 1), (2, 3), (3, 4)])
    assert root_of(t) == 2

    g = nx.DiGraph([(0, 1), (2, 1)])
    import nose.tools
    nose.tools.assert_raises(ValueError, root_of, g)


def is_singly_rooted(s):
    return len(list(roots_of(s))) == 1

Gap = namedtuple('Gap', ['code'])


def is_gap(x):
    return isinstance(x, Gap)


# lowest_common_ancestor : DiGraph x Int x Int -> Int
def lowest_common_ancestor(s, n1, n2):
    s_u = s.to_undirected()
    path = nx.shortest_path(s_u, n1, n2)
    subtree = s.subtree(path)
    return root_of(subtree)


def gaps_under(s, word_id):
    """ Return the indices in the immediate phrase for the gaps under a
    given node. Assumes tree-structured sentence. """
    immediate_phrase = immediate_phrase_of(s, word_id)
    immediate_phrase_set = set(immediate_phrase)
    blocks = blocks_of(s, word_id)
    for i, block in enumerate(blocks):
        if i == 0:
            pass
        else:
            left_endpoint = block[0]
            right_endpoint = block[-1]
            previous_right_endpoint = blocks[i - 1][-1]
            gap_range = range(previous_right_endpoint + 1, left_endpoint)
            subforest = s.subgraph(gap_range)
            left_endpoint_head = left_endpoint
            while left_endpoint_head not in immediate_phrase_set:
                left_endpoint_head = head_of(s, left_endpoint_head)
            i_gap = immediate_phrase.index(left_endpoint_head)
            for root in roots_of(subforest):
                yield i_gap, Gap(classify_gap(s, word_id, root))
            

def classify_gap(s, word_id, root):
    s_u = s.to_undirected()
    path = nx.shortest_path(s_u, word_id, root)
    def gen():
        for n1, n2 in zip(path, path[1:]):
            if heads_of(s, n1) == [n2]:
                yield 'h'
            else:
                yield 'd'
    return "".join(gen())


def test_gaps_under():
    t = nx.DiGraph([(0, 1), (1, 3), (0, 2), (1, 4)]) 
    assert list(gaps_under(t, 0)) == []
    assert list(gaps_under(t, 1)) == [(1, Gap('hd'))]

    t = nx.DiGraph([(4, 3), (3, 1), (4, 2), (4, 0), (3, 5)])
    assert list(gaps_under(t, 4)) == []
    assert list(gaps_under(t, 3)) == [(1, Gap('hd')), (2, Gap('h'))]

    roger_t = nx.DiGraph([
        (0, 1), (0, 2),
        (1, 3), (1, 4),
        (2, 5), (2, 10),
        (3, 6), (3, 7),
        (4, 8), (4, 9)
    ])
    assert list(gaps_under(roger_t, 2)) == [
        (1, Gap('hdd')),
        (1, Gap('hdd')),
        (2, Gap('hddd')),
        (2, Gap('hddd')),
        (2, Gap('hddd')),
        (2, Gap('hddd')),
    ]

    rr_t = nx.DiGraph([(0, 1), (1, 2), (2, 4), (0, 3)])
    assert list(gaps_under(rr_t, 2)) == [(1, Gap('hhd'))]


def immediate_phrase_of(s, word_id, with_gaps=False):
    """ Return the dependents of word_id in s along with word_id. """
    words = dependents_of(s, word_id)
    words.append(word_id)
    words.sort()
    if with_gaps:
        the_gaps = list(gaps_under(s, word_id))
        if the_gaps:
            indices, gaps = zip(*gaps_under(s, word_id))
            return list(insert_multiple(words, indices, gaps))
        else:
            return words
    else:
        return words


def test_immediate_phrase_of():
    t = nx.DiGraph([(2, 0), (2, 1), (2, 3), (3, 4), (3, 5), (5, 6)])
    assert immediate_phrase_of(t, 2) == [0, 1, 2, 3]
    assert immediate_phrase_of(t, 3) == [3, 4, 5]
    assert immediate_phrase_of(t, 5) == [5, 6]
    assert immediate_phrase_of(t, 0) == [0]

    t = nx.DiGraph([(0, 1), (1, 3), (0, 2)])
    assert immediate_phrase_of(t, 0, with_gaps=True) == [0, 1, 2]
    assert immediate_phrase_of(t, 1, with_gaps=True) == [1, Gap('hd'), 3]


def num_words_in_phrase(s, word_id):
    return len(nx.descendants(s, word_id)) + 1


def test_num_words_in_phrase():
    t = nx.DiGraph([(0, 1), (1, 2), (2, 3)])
    assert num_words_in_phrase(t, 0) == 4
    assert num_words_in_phrase(t, 1) == 3
    assert num_words_in_phrase(t, 2) == 2
    assert num_words_in_phrase(t, 3) == 1


# head_of : DiGraph x Int -> Int    
def head_of(s, word_id):
    """ Return the single head of word_id in s. 
    Die if word_id has 0 or more than 1 heads. 
    """
    return the_only(heads_of(s, word_id))


# get_head_of : DiGraph x Int -> Maybe Int
def get_head_of(s, word_id, default=None):
    """ Return the single head of word_id in s, or None if it has no head.
    Die if word_id has more than one head. """
    heads = heads_of(s, word_id)
    if heads:
        return the_only(heads)
    else:
        return default


def test_head_of():
    t = nx.DiGraph([(1, 0), (1, 2), (2, 3)])
    assert head_of(t, 0) == 1
    assert head_of(t, 2) == 1
    assert head_of(t, 3) == 2

    import nose.tools
    nose.tools.assert_raises(ValueError, head_of, t, 1)


def deptype_to_head_of(s, word_id):
    """ Return the dependency type of the arc to word_id from its head in s. """
    h = head_of(s, word_id)
    return s.edge[h][word_id]['deptype']


def test_deptype_to_head_of():
    t = nx.DiGraph([(0, 1), (1, 2), (2, 3)])
    t.edge[2][3] = {'deptype': 'A'}
    t.edge[1][2] = {'deptype': 'B'}
    assert deptype_to_head_of(t, 3) == 'A'
    assert deptype_to_head_of(t, 2) == 'B'


dependents_of = nx.DiGraph.successors # might need to replace for genericity
heads_of = nx.DiGraph.predecessors

Dependents = namedtuple('Dependents', ['left', 'right'])


def left_right_dependents_of(s, word_id):
    ds = sorted(dependents_of(s, word_id))
    middle = bisect.bisect(ds, word_id)
    return Dependents(ds[:middle], ds[middle:])


def left_dependents_of(s, word_id):
    ds = sorted(dependents_of(s, word_id))
    return ds[:bisect.bisect(ds, word_id)] # would linear search be faster?


def test_left_dependents_of():
    t = nx.DiGraph([(2, 1), (1, 0), (2, 5), (5, 3), (5, 4)])
    assert left_dependents_of(t, 2) == [1]
    assert left_dependents_of(t, 1) == [0]
    assert left_dependents_of(t, 5) == [3, 4]
    assert left_dependents_of(t, 0) == []


def right_dependents_of(s, word_id):
    ds = sorted(dependents_of(s, word_id))
    return ds[bisect.bisect(ds, word_id):]


def test_right_dependents_of():
    t = nx.DiGraph([(2, 1), (1, 0), (2, 3), (3, 4), (3, 5)])
    assert right_dependents_of(t, 2) == [3]
    assert right_dependents_of(t, 3) == [4, 5]
    assert right_dependents_of(t, 5) == []
    

is_ancestor = nx.has_path


def is_descendent(s, word_id_1, word_id_2):
    return is_ancestor(s, word_id_2, word_id_1)


def is_tree(s):
    degrees = Counter(degree for n, degree in s.in_degree_iter())
    return set(degrees.keys()) == {0, 1} and degrees[0] == 1 and degrees[1] > 0


def block_endpoints_of(s):
    """ Kuhlmann's (2012: 364-5) O(n) algorithm to identify block endpoints """
    # Comments are from Kuhlmann's paper.
    # We start at a virtual root node which serves as the parent of the real
    # root node. 
    current = None
    marked = {current}
    found_blocks_right = defaultdict(list)
    found_blocks_left = defaultdict(list)
    
    # For each node n in the precedence order of D, we follow the shortest path
    # from the node current to n.
    for node in s.nodes():
        # To determine this path, we compute the lowest common ancestor lca of
        # the two nodes, using a set of markings on the nodes. At the beginning
        # of each iteration of the for loop, all ancestors of current (including
        # the virtual root node) are marked; therefore, we find lca by going
        # upwards from next to the first node that is marked. 
        lca = node
        stack = []  # will contain the path from node to its lca with current
        
        # lca moves up from node to the lowest marked thing
        while lca not in marked:
            stack.append(lca)
            lca = get_head_of(s, lca)
            
        # To restore the loop invariant, we then unmark all nodes on the path
        # from current to lca.
        # current moves up to lca
        while current != lca:
            # move up from current to the parent of current
            marked.remove(current)
            found_blocks_right[current].append(node - 1)
            current = get_head_of(s, current)
            
        # current moves down to node
        while stack:
            current = stack.pop()
            marked.add(current)
            # Each time we move down from a node to one of its children, we
            # record the information that next is the left endpoint of a block
            # of current.
            found_blocks_left[current].append(node)
        assert current == node

    # The while loop takes us from the last node of the dependency tree back to
    # the virtual root node.
    while current is not None:
        # |w| is the right endpoint of a block of current
        # move up from current to the parent of current
        marked.remove(current)
        # Symmetrically, each time we move up from a node to its parents, we
        # record the information that next - 1 is the right endpoint of a block
        # of current.
        found_blocks_right[current].append(node)
        current = get_head_of(s, current)
        
    return dict(found_blocks_left), dict(found_blocks_right)


def blocks_of(s, word_id=None):
    """ Blocks of a dependency tree.
    blocks_of(tree) returns the dict of blocks for all nodes. 
    blocks_of(tree, n) returns the blocks of node n.
    
    Kuhlmann (2013: 363): "For a node u of D, a block of u is the longest 
    segment consisting of descendents of u. This means that the left endpoint of
    a block of u either is the first node in its component, or is preceded by a 
    node that is not a descendent of u. A symmetric property holds for the right
    endpoint." 
    """
    left_endpoints, right_endpoints = block_endpoints_of(s)
    assert left_endpoints.keys() == right_endpoints.keys()
    d = {
        n : [
            list(range(l, r + 1))
            for l, r in zip(left_endpoints[n], right_endpoints[n])
        ]
        for n in left_endpoints.keys()
    }
    if word_id is None:
        return d
    else:
        return d[word_id]


def test_blocks_of():
    # Example from Kuhlmann (2013: 363)
    s = nx.DiGraph([(3, 2), (2, 1), (3, 4), (4, 8), (2, 5), (5, 7), (7, 6)])
    blocks = blocks_of(s)
    assert blocks[2] == [[1, 2], [5, 6, 7]]
    assert blocks_of(s, 2) == blocks[2]


def block_degree(s):
    """ Block degree of a dependency tree (Kuhlmann, 2013) """
    return max(len(blocks) for blocks in blocks_of(s).values())


def gap_degree(s):
    """ Gap degree (block degree - 1) of a sentence. Maximum number of
    discontinuities in phrases of the sentence. """
    return block_degree(s) - 1


def test_gap_degree():
    zero = nx.DiGraph([(0, 1), (1, 2), (2, 3)])
    assert gap_degree(zero) == 0

    one = nx.DiGraph([(0, 1), (1, 3), (0, 2)])
    assert gap_degree(one) == 1

    one2 = nx.DiGraph([(0, 1), (0, 2), (1, 3), (2, 4)])
    assert gap_degree(one2) == 1

    two = nx.DiGraph([(0, 1), (0, 2), (0, 4), (1, 3), (1, 5)])
    assert gap_degree(two) == 2


def is_well_nested(s):
    """ Check if a sentence is well-nested in the sense of Kuhlmann (2013). """
    raise NotImplementedError


def _test_is_well_nested():
    wn = nx.DiGraph([(0, 1), (0, 2), (0, 4), (1, 3), (1, 5)])
    assert is_well_nested(wn)
    
    nwn = nx.DiGraph([(0, 1), (1, 3), (0, 2), (2, 4)])
    assert not is_well_nested(nwn)


def is_monotonic(cmp, xs):
    try:
        return all(cmp(x, y) for x, y in sliding(xs, 2))
    except StopIteration: # TODO fix sliding so this doesn't need to be special
        return True


def test_is_monotonic():
    mle = [-1, 2, 3, 4, 4]
    assert is_monotonic(operator.le, mle)
    assert not is_monotonic(operator.lt, mle)
    assert not is_monotonic(operator.gt, mle)
    assert not is_monotonic(operator.ge, mle)

    ml = [-100, 1, 3, 4]
    assert is_monotonic(operator.le, ml)
    assert is_monotonic(operator.lt, ml)
    assert not is_monotonic(operator.gt, ml)
    assert not is_monotonic(operator.ge, ml)

    mge = [4, 4, 3, 2, -1]
    assert not is_monotonic(operator.le, mge)
    assert not is_monotonic(operator.lt, mge)
    assert not is_monotonic(operator.gt, mge)
    assert is_monotonic(operator.ge, mge)

    mg = [4, 3, 1, -100]
    assert not is_monotonic(operator.le, mg)
    assert not is_monotonic(operator.lt, mg)
    assert is_monotonic(operator.gt, mg)
    assert is_monotonic(operator.ge, mg)

    nm = [1, 2, -1, -2]
    assert not is_monotonic(operator.le, nm)
    assert not is_monotonic(operator.lt, nm)
    assert not is_monotonic(operator.gt, nm)
    assert not is_monotonic(operator.ge, nm)

    assert is_monotonic(lambda x, y: shouldnt_be_evaluated, [])


def immediate_phrase_has_monotonic_ordering(s, n, left_cmp, right_cmp):
    left = left_dependents_of(s, n)
    right = right_dependents_of(s, n)
    return (
        is_monotonic(left_cmp, (num_words_in_phrase(s, x) for x in left))
        and is_monotonic(right_cmp, (num_words_in_phrase(s, x) for x in right))
    )


def immediate_phrase_has_outward_ordering(s, n):
    return immediate_phrase_has_monotonic_ordering(
        s,
        n,
        operator.ge,
        operator.le
    )


def has_monotonic_ordering(s, left_cmp, right_cmp):
    return all(
        immediate_phrase_has_monotonic_ordering(s, n, left_cmp, right_cmp)
        for n in s.nodes_iter()
    )


def has_outward_ordering(s):
    return has_monotonic_ordering(s, operator.ge, operator.le)


def has_pseudo_outward_ordering(s):
    def conditions():
        for node in s.nodes_iter():
            left = left_dependents_of(s, node)
            if left:
                first_weight = num_words_in_phrase(s, left[0])
                yield all(
                    first_weight >= num_words_in_phrase(s, n)
                    for n in left[1:]
                )
            right = right_dependents_of(s, node)
            if right:
                last_weight = num_words_in_phrase(s, right[-1])
                yield all(
                    last_weight >= num_words_in_phrase(s, n)
                    for n in right[:-1]
                )
    return all(conditions())


def test_has_outward_ordering():
    good_left_edges = [(6, 5), (6, 4), (4, 3), (6, 2), (2, 1), (1, 0)]
    good_right_edges = [(7, 8), (7, 9), (9, 10), (7, 11), (11, 12), (12, 13)]
    
    tree = nx.DiGraph(good_left_edges)
    assert has_outward_ordering(tree)

    tree = nx.DiGraph(good_left_edges[:-2])
    assert not has_outward_ordering(tree)
    
    tree = nx.DiGraph(good_right_edges)
    assert has_outward_ordering(tree)

    tree = nx.DiGraph(good_right_edges[:-2])
    assert not has_outward_ordering(tree)

    tree = nx.DiGraph(good_left_edges + good_right_edges)
    assert has_outward_ordering(tree)


def test_has_monotonic_ordering():
    good_left_edges = [(6, 5), (6, 4), (4, 3), (6, 2), (2, 1), (1, 0)]
    good_right_edges = [(7, 8), (7, 9), (9, 10), (7, 11), (11, 12), (12, 13)]    
    bad_right_edges = [(7, 8), (8, 9), (9, 10), (7, 11), (11, 12), (7, 13)]
    bad_left_edges = [(6, 5), (5, 4), (4, 3), (6, 2), (2, 1), (6, 0)]

    tree = nx.DiGraph(good_right_edges)
    assert not has_monotonic_ordering(tree, operator.le, operator.ge)    
    
    tree = nx.DiGraph(bad_right_edges)
    assert has_monotonic_ordering(tree, operator.le, operator.ge)

    tree = nx.DiGraph(good_left_edges)
    assert not has_monotonic_ordering(tree, operator.le, operator.ge)

    tree = nx.DiGraph(bad_left_edges)
    assert has_monotonic_ordering(tree, operator.le, operator.ge)

    tree = nx.DiGraph(bad_left_edges + bad_right_edges)
    assert has_monotonic_ordering(tree, operator.le, operator.ge)
    assert not has_monotonic_ordering(tree, operator.ge, operator.le)


def insert_multiple(xs, indices, values):
    indices = set(indices)
    values_it = iter(values)
    for i, x in enumerate(xs):
        if i in indices:
            yield next(values_it)
            indices.remove(i)
        yield x
    for i_left_over in indices:
        yield next(values_it)


def crossings_in(tree):
    for edge in tree.edges():
        n1, n2 = sorted(edge)
        for edge_ in tree.edges():
            n1_, n2_ = sorted(edge_)
            if not (n2_ <= n1
                    or n2 <= n1_
                    or (n1 <= n1_ and n2_ <= n2)
                    or (n1_ <= n1 and n2 <= n2_)):
                yield frozenset({edge, edge_})


def num_crossings_in(tree):
    return len(set(crossings_in(tree)))


def edge_projective(graph, arc):
    """ determine if the given edge is part of a projective graph.
    projective iff given n1 and n2 there is no edge containing n1' and n2' such
    that n1 < n1' < n2 < n2' or n1 > n1' > n2 > n2'. """
    n1, n2 = sorted(arc) # now we know n1 < n2
    def conditions():
        for arc in graph.edges():
            n1_, n2_ = sorted(arc) # now we know n1_ < n2_
            yield (n2_ <= n1
                   or n2 <= n1_
                   or (n1 <= n1_ and n2_ <= n2)
                   or (n1_ <= n1 and n2 <= n2_))
    return all(conditions())


def is_projective(graph):
    return all(edge_projective(graph, edge) for edge in graph.edges())


def gaps_left_right(s, h):
    phrase = immediate_phrase_of(s, h, with_gaps=True)
    h_index = phrase.index(h)
    for i, x in enumerate(phrase):
        if is_gap(x):
            yield s, h, i < h_index, x.code


def is_projective_on_left(s):
    for n in s.nodes():
        gaps = gaps_left_right(s, n)
        if any(gap_on_left for _, _, gap_on_left, _ in gaps):
            return False
    return True


def is_projective_on_right(s):
    for n in s.nodes():
        gaps = gaps_left_right(s, n)
        if any(not gap_on_left for _, _, gap_on_left, _ in gaps):
            return False
    return True


def transitive_head_of(s, n, k):
    assert k >= 0
    while k > 0:
        n = head_of(s, n)
        k -= 1
    return n


def transitive_heads(s, n):
    while True:
        yield n
        n = head_of(s, n)
        if n == 0:
            break


if __name__ == '__main__':
    import nose
    nose.runmodule()

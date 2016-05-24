#!/usr/bin/python3
from collections import namedtuple
import depgraph

def Head(value):
    return ('Head', value)

def Ancestor(degree, value):
    return ('Ancestor', degree, value)

def Dependent(value):
    return ('Dependent', value)

# Some combinators for building up conditioning functions...

def f_name(f):
    try:
        return f.__qualname__
    except AttributeError:
        return f.__name__

def phrase_f(f):
    result = lambda sentence, phrase, h: [f(sentence, n, h) for n in phrase]
    name = "PHRASE_XX_%s_XX" % f_name(f)
    result.__name__ = name
    result.__qualname__ = name
    return result

def dependent_f(h_f, d_f, both_f):
    def result(sentence, d):
        h = depgraph.head_of(sentence, d)
        return (h_f(sentence, h), d_f(sentence, d), both_f(sentence, h, d))
    name = "dependent_f(%s)" % ",".join(map(f_name, [h_f, d_f, both_f]))
    result.__name__ = name
    result.__qualname__ = name
    return result

def edge_f(h_f, d_f, both_f):
    def result(sentence, h, d):
        return (h_f(sentence, h), d_f(sentence, d), both_f(sentence, h, d))
    name = "edge_f(%s)" % ",".join(map(f_name, [h_f, d_f, both_f]))
    result.__name__ = name
    result.__qualname__ = name
    return result    

def get_edge_dt(sentence, h, d):
    return sentence[h][d]['deptype']

def nothing(*args):
    return None

# product_f :: *[*A -> b] -> *A -> [b]
def product_f(*fs):
    result = lambda *args: tuple(f(*args) for f in fs)
    name = "PRODUCT_XX_%s_XX" % "_".join(f_name(f) for f in fs)
    result.__name__ = name
    result.__qualname__ = name
    return result

def head_dep_f(head_f, dep_f):
    def f(sentence, n, h):
        if n == h:
            return Head(head_f(sentence, n))
        else:
            return Dependent(dep_f(sentence, n))
    name = "HEAD_DEP_XX_%s_%s_XX" % (f_name(head_f), f_name(dep_f))
    f.__name__ = name
    f.__qualname__ = name
    return f

def get_dt(sentence, n):
    if n == 0:
        return 'TOP'
    else:
        return depgraph.deptype_to_head_of(sentence, n)

def get_attr(attr):
    def getter(sentence, n):
        return sentence.node[n].get(attr)
    return getter

get_pos = get_attr('pos')
get_word = get_attr('word')

def get_pos2(sentence, n):
    pos2 = sentence.node[n].get('pos2')
    if pos2 is None or pos2 == "_":
        return get_attr('pos')(sentence, n)
    else:
        return pos2

dep_dt = phrase_f(head_dep_f(nothing, get_dt))
dep_dt_and_pos = phrase_f(head_dep_f(nothing, product_f(get_dt, get_pos)))
dep_dt_and_pos2 = phrase_f(head_dep_f(nothing, product_f(get_dt, get_pos2)))

head_pos2_dep_dt_and_pos2 = phrase_f(
    head_dep_f(
        get_pos2,
        product_f(get_dt, get_pos2)
        )
    )

head_dt_and_pos_dep_dt_and_pos = phrase_f(
    head_dep_f(
        product_f(get_dt, get_pos),
        product_f(get_dt, get_pos)
        )
    )

head_pos_dep_dt_and_pos = phrase_f(
    head_dep_f(
        get_pos,
        product_f(get_dt, get_pos)
        )
    )

full_cond = phrase_f(
    head_dep_f(
        product_f(get_dt, get_pos),
        product_f(get_dt, get_pos)
        )
    )

full_cond2 = phrase_f(
    head_dep_f(
        product_f(get_dt, get_pos2),
        product_f(get_dt, get_pos2)
        )
    )

ROOT_HEAD = Head(value=None)

def is_head(x):
    return x[0] == 'Head'

def head_index(xs):
    for i, x in enumerate(xs):
        if is_head(x):
            return i
    raise ValueError("No head found in phrase %s" % xs)

if __name__ == '__main__':
    import nose
    nose.runmodule()


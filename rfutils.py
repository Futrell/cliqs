import functools
import itertools as it

_SENTINEL = object()

def sliding(iterable, n):
    """ Sliding

    Yield adjacent elements from an iterable in a sliding window
    of size n.

    Parameters:
        iterable: Any iterable.
        n: Window size, an integer.

    Yields:
        Tuples of size n.

    Example:
        >>> lst = ['a', 'b', 'c', 'd', 'e']
        >>> list(sliding(lst, 2))
        [('a', 'b'), ('b', 'c'), ('c', 'd'), ('d', 'e')]

    """
    its = it.tee(iterable, n)
    for i, iterator in enumerate(its):
        for _ in range(i):
            next(iterator)
    return zip(*its)

def the_only(xs):
    first_time = True
    x = _SENTINEL
    for x in xs:
        if first_time:
            first_time = False
        else:
            raise ValueError("Iterable passed to the_only had second value %s" % x)
    if x is _SENTINEL:
        raise ValueError("Empty iterable passed to the_only")
    else:
        return x

def mean(xs):
    """ Mean of elements in an iterable. """
    total = 0
    n = 0
    for x in xs:
        total += x
        n += 1
    try:
        return total/n
    except ZeroDivisionError:
        return 0.0

def compose(*fs):
    return functools.reduce(lambda f, g: lambda x: f(g(x)), fs)

    
def conditional_counts(pairs):
    """ Given an iterable of pairs (X, Y), produce a dict X -> Y -> Int with the
    conditional counts of values of values of Y conditional on values of X. """
    if isinstance(pairs, dict):
        pairs = pairs.items()
    d = {}
    for x, y in pairs:
        if x in d:
            d[x][y] += 1
        else:
            d[x] = Counter({y: 1})
    return d


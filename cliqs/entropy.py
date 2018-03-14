from __future__ import division
from math import log, log2
from collections import Counter, defaultdict

base = log(2)

def entropy(counts):
    """ entropy

    Fast calculation of Shannon entropy from an iterable of positive numbers. 
    Numbers are normalized to form a probability distribution, then entropy is
    computed.

    Generators are welcome.

    Params:
        counts: An iterable of positive numbers.

    Returns:
        Entropy of the counts, a positive number.

    """
    if isinstance(counts, Counter):
        counts = counts.values()
    total = 0.0
    clogc = 0.0
    for c in counts:
        total += c
        try:
            clogc += c * log(c)
        except ValueError:
            pass
    try:
        return -(clogc/total - log(total)) / base
    except ZeroDivisionError:
        return 0.0

def surprisals(counts):
    if isinstance(counts, Counter):
        counts = counts.items()
    counts = list(counts)
    A = log2(sum(c for _, c in counts))
    for x, count in counts:
        yield x, -log2(count) + A

def conditional_entropy(iterable):
    if isinstance(iterable, dict):
        return conditional_entropy_of_counters(iterable)
    else:
        counts = defaultdict(Counter)
        for x, y in pairs:
            counts[y][x] += 1
        conditional_counts = (counts_in_context.values()
                              for counts_in_context in counts.values())
        return entropy.conditional_entropy_of_counts(conditional_counts)

def custom_conditional_entropy(dict_of_counters, entropy_f):
    grand_total = 0
    result = 0
    for counter in dict_of_counters.values():
        total = sum(counter.values())
        grand_total += total
        result += total * entropy_f(counter.values())
    return result / grand_total

def conditional_entropy_of_counts(iterable_of_iterables):
    """ conditional entropy of counts

    Conditional entropy of a conditional distribution represented as an 
    iterable of iterables of counts.

    """
    entropy = 0.0
    grand_total = 0.0
    for counts in iterable_of_iterables:
        total = 0.0
        clogc = 0.0
        for c in counts:
            total += c
            try:
                clogc += c * log(c)
            except ValueError:
                pass
        grand_total += total
        entropy += total * -(clogc/total - log(total)) / base
    return entropy / grand_total

def conditional_entropy_of_counters(dict_of_counters):
    conditional_counts = (counter.values()
                          for counter in dict_of_counters.values())
    return conditional_entropy_of_counts(conditional_counts)

def mutual_information_from_pairs(pairs):
    return mutual_information(Counter(pairs))

def pointwise_mutual_informations(counts):
    if isinstance(counts, dict):
        counts = counts.items()
    Z = 0
    c_x = Counter()
    c_y = Counter()
    for (x, y), c_xy in counts:
        c_x[x] += c_xy
        c_y[y] += c_xy
        Z += c_xy
    A = log2(Z)
    h_x = dict(surprisals(c_x))
    h_y = dict(surprisals(c_y))
    for (x, y), c_xy in counts:
        yield (x, y), h_x[x] + h_y[y] + log2(c_xy) - A

def test_pointwise_mutual_informations():
    xs = pointwise_mutual_informations(Counter(['aa', 'ab', 'ba', 'bb']))
    assert all(m == 0 for _, m in xs)
    xs = pointwise_mutual_informations(Counter(['aa', 'bb']))
    assert all(m == 1 for _, m in xs)
        
def mutual_information(counts):
    """ mutual information
    
    Takes iterable of tuples of form ((x_value, y_value), count)
    or counter whose keys are tuples (x_value, y_value).

    Stores marginal counts in memory.

    """
    if isinstance(counts, dict):
        counts = counts.items()

    total = 0
    c_x = Counter()
    c_y = Counter()
    clogc = 0

    for (x, y), c_xy in counts:
        total += c_xy
        c_x[x] += c_xy
        c_y[y] += c_xy
        try:
            clogc += c_xy * log(c_xy)
        except ValueError:
            pass

    try:
        return (log(total)
                + (clogc
                - sum(c*log(c) for c in c_x.values())
                - sum(c*log(c) for c in c_y.values()))
                / total) / base
    except ZeroDivisionError:
        return 0.0
    except ValueError:
        return 0.0

def _generate_counts(lines):
    first_line = next(lines)
    first_line_elems = first_line.split()
    if any(x.isdigit() for x in first_line_elems):
        yield _get_count(first_line)
        for line in lines:
            yield _get_count(line)
    counts = Counter(lines)
    counts[line] += 1
    for count in counts.values():
        yield count
      
def _get_count(line):
    line = line.split()
    for x in line:
        if x.isdigit():
            return float(x)

if __name__ == "__main__":
    import sys
    lines = sys.stdin
    result = entropy(_generate_counts(lines))
    print(result)

from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from cliqs.compat import *

import sys
import copy
import itertools
import functools
import random
import csv
import pickle

import rfutils
import networkx as nx
#from distributed import Executor, as_completed

import cliqs.mindep as mindep
import cliqs.opt_mindep as opt_mindep
import cliqs.linearize as lin
import cliqs.corpora as corpora

OPTS = {
    'fix_content_head': False,
    'collapse_flat': True,
    'verbose': True,
    'remove_fw': False,
    'remove_punct': True,
}
LOWER_LENGTH_LIMIT = 3

def load_all_corpora_into_memory(corpora):
    for corpus in corpora:
        corpus.load_into_memory()
        
# load_all_corpora_into_memory(corpora.corpora)     

def with_open(filename, mode, f):
    with open(filename, mode) as infile:
        return f(infile)

MODEL_FILENAME_TEMPLATE = "models/%s_%s.dill"
CONDITIONING = opt_mindep.get_deptype

@rfutils.memoize
def load_linearization_model(lang, spec):
    return with_open(MODEL_FILENAME_TEMPLATE % (lang, spec), 'rb', pickle.load)

# Make a dataframe from applying deterministic functions to each sentence, and
# from applying random sampling functions to each sentence NUM_RANDOM_SAMPLES
# times. Each column is the result of applying a certain function or a certain
# sample from applying a certain random function.

# E.g., columns are lang, length, det_f_a, det_f_b, random_f_a_1, random_f_a_2,
# ..., random_f_a_100, ...

# In principle it seems it should be faster to apply parallelism here at the
# level of sentences, hence the parallel flag for this function. But in practice
# it looks like that's actually slower than parallelizing over corpora, for some
# reason. Maybe with more cpu-intensive linearization procedures there would be
# gains from sentence-level parallelism?

NUM_RANDOM_SAMPLES = 100

def generate_rows(sentences,
                  lang,
                  deterministic_fns,
                  random_fns,
                  parallel=False):
    def gen_row(i_s):
        i, s = i_s
        s = corpora.DepSentence.from_digraph(s)
        def gen():
            yield 'start_line', i
            yield 'lang', lang
            yield 'length', len(s.nodes())
            for det_fn_name, det_fn in deterministic_fns.items():
                yield det_fn_name, det_fn(s, lang)
            for rand_fn_name, rand_fn in random_fns.items():
                for j in range(NUM_RANDOM_SAMPLES):
                    yield (
                        "%s%s" % (rand_fn_name, j),
                        rand_fn(s, lang, j, deptypes)
                    )
        return dict(gen())
    if any('per_lang' in k for k in random_fns.keys()):
        sentences = list(sentences) # I don't see any other way
        deptypes = frozenset({
            CONDITIONING(sentence, edge) for sentence in sentences
            for edge in sentence.edges_iter(data=True)
        })
    else:
        deptypes = None
        
    if parallel:
        die
        rows = pmap(gen_row, enumerate(sentences))
    else:
        # for some obscure reason, when this function is run as part of a 
        # top-level pmap, you have to do the mapping over sentences like this, 
        # lest you get pickling errors. 
        def gen_rows():
            for i, s in enumerate(sentences):
                if len(s) > LOWER_LENGTH_LIMIT:
                    yield gen_row((i, s))
        rows = gen_rows()
    return rows

# Reduction functions

NA = float('nan')

def make_reduction_f(r):
    def reduction_f(f):
        @functools.wraps(f)
        def wrapper(s, *a, **k):
            try:
                result = f(s, *a, **k)
            except Exception as e:
                print(e, file=sys.stderr)
                return NA
            return r(s, linearization=result)
        return wrapper
    return reduction_f

deplen_f = make_reduction_f(mindep.deplen)
max_embedding_depth_f = make_reduction_f(mindep.max_embedding_depth)
sum_embedding_depth_f = make_reduction_f(mindep.sum_embedding_depth)

def deptypes(s):
    for h, d in s.edges():
        if h != 0:
            yield s.edge[h][d]['deptype']

deplens_f = make_reduction_f(mindep.deplens)
deptypes_f = make_reduction_f(deptypes)

# Deterministic dep len functions

def identity(x, *_):
    return x

real_deplen = deplen_f(identity)
real_sum_embedding_depth = sum_embedding_depth_f(identity)
real_max_embedding_depth = max_embedding_depth_f(identity)

def real_deplen(s, *_): # keep
    return mindep.deplen(s)

def mhd(s, *_):
    """ Mean Heirarchical Distance from Yingqi Jing's presentation at DepLing """
    return mean(len(list(path_to_root(s, n))) for n in s.nodes())

def path_to_root(s, n):
    while s.in_edges(n):
        h = s.head_of(n)
        yield h
        n = h
            
def real_best_case_memory_cost(s, *_):
    return mindep.best_case_memory_cost(s)

def real_deplen_filtered(*filters):
    filters = list(filters)
    def deplen(s, *_):
        return mindep.deplen(s, filters=filters)
    return deplen

def min_deplen(s, *_): 
    min_deplen, min_deplin = mindep.mindep_projective_alternating(s)
    return min_deplen

def min_deplen_opt(**kwds):
    def md(s, *_):
        min_deplen, _ = mindep.mindep_projective_alternating(s, **kwds)
        return min_deplen
    return md

def min_deplen_filtered(*filters):
    filters = list(filters)
    def deplen(s, *_):
        _, min_deplin = mindep.mindep_projective_alternating(s)
        return mindep.deplen(s, linearization=min_deplin, filters=filters)
    return deplen

def ordered_deplen(s, *_):
    result, _ = mindep.linearize_by_weight_head_final(s)
    return result

def weighted_deplen(s, lang, *_): 
    weights = WEIGHTS[lang]
    lin = opt_mindep.get_linearization(s, weights, thing_fn=CONDITIONING)
    score = mindep.deplen(s, lin)
    return score

# Random dep len functions

def deplen_random_sample_nobias_filtered(*filters):
    filters = list(filters)
    def deplen(s, *_):
        lin = mindep.randlin_projective(s, head_final_bias=0)[1]
        return mindep.deplen(s, linearization=lin, filters=filters)
    return deplen

def random_sample_nobias(s, *_):
    return mindep.randlin_projective(s)[-1]

def random_sample_opt(**kwds):
    def rs(s, *_):
        return mindep.randlin_projective(s, **kwds)[0]
    return rs

#random_sample_nobias = random_sample_opt(head_final_bias=0)
random_sample_headfinal = random_sample_opt(head_final_bias=1)

def random_sample_best_case_memory_cost(s, *_):
    _, lin = mindep.randlin_projective(s)
    return mindep.best_case_memory_cost(s, linearization=lin)
        
def random_sample_weighted(s, *_): # redo these using WeightedLin class
    return opt_mindep.randlin_fixed_weights(s, thing_fn=CONDITIONING, head_final=False)[0]

@rfutils.memoize
def get_weights(lang, i, deptypes, head_final):
    weights = opt_mindep.rand_fixed_weights(deptypes, head_final=head_final)
    return weights

def random_sample_weighted_per_lang(s, lang, i, deptypes):
    weights = get_weights(lang, i, deptypes, False)
    return opt_mindep.randlin_from_weights(s, weights, CONDITIONING)[0]

def random_sample_weighted_headfinal_per_lang(s, lang, i, deptypes):
    weights = get_weights(lang, i, deptypes, True)
    return opt_mindep.randlin_from_weights(s, weights, opt_mindep.get_deptype)[0]

def random_sample_weighted_headfinal(s, *_):
    return opt_mindep.randlin_fixed_weights(s, thing_fn=opt_mindep.get_deptype, head_final=True)[0]

def random_sample_weighted_best_case_memory_cost(s, *_):
    _, lin = opt_mindep.randlin_fixed_weights(s)
    return mindep.best_case_memory_cost(s, linearization=lin)

def random_sample_fullyfree(s, *_):
    lin = [n for n in s.nodes() if n != 0]
    random.shuffle(lin)
    return mindep.deplen(s, linearization=[0] + lin)

def random_sample_fullyfree_headfinal(s, *_):
    # still broken...
    def shift_right_of_descendants(xs, x):
        i = xs.index(x)
        descendants = closure[x]
        indices = [i for i, x in enumerate(xs) if x in descendants]
        if not indices:
            return xs # no change necessary
        right_boundary = indices[-1] + 1
        if right_boundary <= i: # CAREFUL
            return xs # no change necessary
        else:
            return itertools.chain(
                splice(xs[:right_boundary], i),
                [x],
                xs[right_boundary:]
            )
    # make a random linearization
    lin = [n for n in s.nodes() if n != 0]
    random.shuffle(lin)
    closure = [nx.descendants(s, n) for n in sorted(s.nodes())]
    # shift each word to be to the right of its transitive descendents
    for x in lin:
        lin = list(shift_right_of_descendants(lin, x))
    assert sorted(lin) == list(range(1, len(s)))
    return lin
    
def splice(xs, i):
    for	j, x in	enumerate(xs):
        if j !=	i:
            yield x    

# Model-based functions

def random_sample_proj_lin_spec(spec):
    def random_sample_proj_lin(s, lang, *_):
        m = load_linearization_model(lang, spec)
        return lin.proj_lin(m, s)
    return random_sample_proj_lin

# Some filters to be used with the *_filtered functions above

def negate(f):
    return lambda *args: not f(*args)

def is_medial(sentence, lin, hd):
    h, d = hd
    nodes = [d for _, d in sentence.out_edges_iter(h)]
    nodes.append(h)
    nodes.sort()
    return nodes[0] == h or nodes[-1] == h

not_medial = negate(is_medial)

def only_left(sentence, lin, hd):
    h, d = hd
    return lin[d] < lin[h]

def filter_edges(s, filters):
    s = copy.deepcopy(s)
    lin = {n:n for n in s.node.keys()}
    for edge in s.edges():
        if not all(edge[0] == 0 or f(s, lin, edge) for f in filters):
            s.remove_edge(*edge)
    return s

def build_it(lang, corpora=corpora.ud_corpora, parallel=False):
    return generate_rows(
        corpora[lang].sentences(**OPTS),
        lang,
        {
            'deplen': real_deplen,
            #'max_depth': max_embedding_depth_f(identity),
            #'sum_depth': sum_embedding_depth_f(identity),
            #'bcmc': real_best_case_memory_cost,
            #'min_deplen_headfixed': min_deplen_opt(move_head=False),
            #'min_deplen': min_deplen,
            #'min_deplen_headfinal': ordered_deplen,
            #'mhd': mhd,
        },
        {
            'rand_deplen': deplen_f(random_sample_nobias),
            #'rand_max_depth': max_embedding_depth_f(random_sample_nobias),
            #'rand_sum_depth': sum_embedding_depth_f(random_sample_nobias),
            
            #'rand_proj_lin_r_lic': deplen_f(random_sample_proj_lin_spec('r|lic')),
            #'rand_proj_lin_dr_lic': deplen_f(random_sample_proj_lin_spec('dr|lic')),
            #'rand_proj_lin_hdr_lic': deplen_f(random_sample_proj_lin_spec('hdr|lic')),
            
            #'rand_proj_lin_r_mle': deplen_f(random_sample_proj_lin_spec('r|moo')),
            #'rand_proj_lin_dr_mle': deplen_f(random_sample_proj_lin_spec('dr|moo')),
            #'rand_proj_lin_hdr_mle': deplen_f(random_sample_proj_lin_spec('hdr|moo')),
            
            #'rand_proj_lin_perplex': deplen_f(random_sample_proj_lin_spec('hdr+r|oo+n123')),
            #'rand_proj_lin_acceptable': deplen_f(random_sample_proj_lin_spec('hdr|n123')),
            #'rand_proj_lin_meaningsame': deplen_f(random_sample_proj_lin_spec('hdr|n3')),
            
            #'rand_bcmc': random_sample_best_case_memory_cost,
            #'rand_deplen_fixed': random_sample_weighted, # missing lin
            'rand_deplen_fixed_per_lang': random_sample_weighted_per_lang,
            #'rand_weight_bcmc': random_sample_weighted_best_case_memory_cost,
            #'rand_deplen_headfinal': random_sample_headfinal, 
            #'rand_deplen_headfinal_fixed_per_lang': random_sample_weighted_headfinal_per_lang,
            #'rand_known_order': random_sample_known_order,
            #'rand_deplen_headfixed': random_sample_opt(move_head=False),
            #'rand_deplen_fullyfree': random_sample_fullyfree,
            #'rand_deplen_fullyfree_headfinal': deplen_f(random_sample_fullyfree_headfinal),
        },
        parallel=parallel,
    )

def dictplus(d1, d2):
    d = d1.copy()
    d.update(d2)
    return d

def imelt(ds, id_vars):
    for d in ds:
        id_d = {key:line[key] for key in id_vars}
        for column, value in d.items():
            if column not in id_vars:
                yield dictplus(id_d, {'variable': column, 'value': value})

def ipostprocess(df, id_vars):
    ds = map(pd.Series.to_dict, df.iterrows())
    melted = imelt(ds, id_vars)
    for d in melted:
        d['real'] = name_fn(d['variable'])
        del d['variable']
        yield d

def postprocess(df):
    import pandas as pd
    dfm = pd.melt(df, id_vars='lang length start_line'.split())
    dfm['real'] = dfm['variable'].map(name_fn)
    del dfm['variable']
    return dfm

def name_fn(var):
    d = [
        ('rand_deplen_headfixed', 'free head-fixed random'),
        ('rand_deplen_headfinal_fixed_per_lang', 'fixed head-consistent random per language'),        
        ('rand_deplen_headfinal_fixed', 'fixed head-consistent random'),
        ('rand_deplen_headfinal', 'free head-consistent random'),              
        ('rand_bcmc_fixed', 'fixed random bcmc'),        
        ('rand_bcmc', 'free random bcmc'),
        ('rand_deplen_fixed_per_lang', 'fixed random per language'),
        ('rand_deplen_fixed', 'fixed random'),
        ('rand_deplen_fullyfree_headfinal', 'nonprojective free head-consistent random'),        
        ('rand_deplen_fullyfree', 'nonprojective free random'),
        ('rand_deplen_gd', 'gd random'),
        ('rand_deplen', 'free random'),        
        ('rand_known_order', 'known random'),
        ('min_deplen_headfixed', 'free head-fixed optimal'),
        ('min_deplen_headfinal', 'free head-consistent optimal'),        
        ('min_deplen_fixed', 'fixed optimal'),        
        ('min_deplen', 'free optimal'),
        ('deplen', 'real'),
        ('bcmc', 'real bcmc'),
        ('mhd', 'mhd'),
     ]
    for prefix, result in d:
        if var.startswith(prefix):
            return result
    else:
        return "".join(c for c in var if not c.isdigit())

    
#executor = Executor()

def pmap(f, xs):
    for future in as_completed(executor.map(f, xs)):
        yield future.result()

def tap(x):
    print(x, file=sys.stderr)
    return x

def main(cmd, *args):
    if cmd == "run":
        langs = tuple(args)
        if langs == ('ud',):
            langs = corpora.ud_langs
            the_corpora = corpora.ud_corpora
        elif langs == ('pud',):
            langs = corpora.pud_langs
            the_corpora = corpora.parallel_corpora
        if langs[0] == 'gd':
            seed = int(langs[1])
            langs = corpora.ud_langs
            the_corpora = corpora.gd_corpora[seed]
        rows = rfutils.flat(
            build_it(lang, parallel=False, corpora=the_corpora)
            for lang in langs
        )
        first_row = rfutils.first(rows)
        writer = csv.DictWriter(sys.stdout, first_row.keys())
        writer.writeheader()
        writer.writerow(first_row)
        for row in rows:
            writer.writerow(row)
    elif cmd == "postprocess":
        import pandas as pd
        filenames = args
        dfs = map(lambda f: pd.read_csv(f, skipfooter=1), filenames)
        rows = itertools.chain(
            ipostprocess(df, "lang length start_line".split())
            for df in dfs
        )
        first_row = rfutils.first(row)
        writer = csv.DictWriter(sys.stdout, first_row.keys())
        writer.writeheader()
        writer.writerow(first_row)
        for row in rows:
            writer.writerow(row)
        #df = functoosl.reduce(pd.DataFrame.append, map(pd.read_csv, filenames))
        #new_df = postprocess(df)
        #new_df.to_csv(sys.stdout)
    else:
        rfutils.err("Unknown command: %s" % cmd)
        
if __name__ == '__main__':
    main(*sys.argv[1:])
    

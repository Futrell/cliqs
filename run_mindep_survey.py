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
#from distributed import Executor, as_completed

import cliqs.mindep as mindep
import cliqs.opt_mindep as opt_mindep
import cliqs.linearize as lin
import cliqs.corpora as corpora
import cliqs.conditioning as cond

NA = float('nan')

OPTS = {'fix_content_head': False, 'collapse_flat': True}

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

LANGS = set(corpora.ud_corpora.keys())

NUM_RANDOM_SAMPLES = 100

def generate_rows(sentences,
                  lang,
                  lin_fs,
                  depvar_fs, # applied to real lin and artificial lins
                  background_depvar_fs, # applied only to real lin
                  parallel=False):
    def process_sentence(i_s):
        i, s = i_s
        s = corpora.DepSentence.from_digraph(s)
        length = len(s) - 1
        # All depvars must return the same length iterable
        # Each row is for one depvar result, across multiple linearizations

        lins = {
            lin_f_name : list(lin_f(s, lang))
            for lin_f_name, lin_f in sorted(lin_fs.items())
        }
        lins['real'] = list(identity_lin(s))

        background_results = [
            [(depvar_f_name, value) for value in depvar_f(s)]
            for depvar_f_name, depvar_f in background_depvar_fs.items()
        ]
            
        results = [
            [(j, lin_f_name, depvar_f_name, value) for value in depvar_f(s, lin)]
            for depvar_f_name, depvar_f in depvar_fs.items()
            for lin_f_name, the_lins in lins.items()
            for j, lin in enumerate(the_lins)
        ]

        bg_rows = zip(*background_results)
        rows = zip(*results)
        
        for k, (bg_results, results) in enumerate(zip(bg_rows, rows)):
            def gen():
                yield 'lang', lang
                yield 'sentence_no', i
                yield 'result_no', k
                yield 'length', length
                for j, lin_f_name, depvar_f_name, value in results:
                    yield '%s_%s_%d' % (lin_f_name, depvar_f_name, j), value
                for depvar_f_name, value in bg_results:
                    yield depvar_f_name, value
                
                
            yield dict(gen())
    if not parallel:
        pmap = map
    return rfutils.flat(pmap(process_sentence, enumerate(sentences)))

def deptypes(s, *_):
    for h, d in s.edges():
        if h != 0:
            yield s.edge[h][d]['deptype']

def get_dep_attrs(f):
    def g(s, *_):
        for h, d in s.edges():
            if h != 0:
                yield f(s, d)
    return g

def get_head_attrs(f):
    def g(s, *_):
        for h, d in s.edges():
            if h != 0:
                yield f(s, h)
    return g

def path_to_root(s, n):
    while s.in_edges(n):
        h = s.head_of(n)
        yield h
        n = h

def deterministic(f):
    @functools.wraps(f)
    def wrapper(*a, **k):
        yield f(*a, **k)
    return wrapper

def sampler(f):
    @functools.wraps(f)
    def wrapper(*a, **k):
        for _ in range(NUM_RANDOM_SAMPLES):
            yield f(*a, **k)
    return wrapper

# deterministic lin functions

@deterministic
def identity_lin(s, *_):
    return range(len(s))

@deterministic
def min_proj_lin(s, *_):
    return mindep.mindep_projective_alternating(s)[-1]

@deterministic
def ordered_lin(s, *_):
    _, result = mindep.linearize_by_weight_head_final(s)
    return result

@deterministic
def weighted_lin(s, lang, *_): 
    weights = WEIGHTS[lang]
    lin = opt_mindep.get_linearization(s, weights, thing_fn=CONDITIONING)
    return lin

@sampler
def rand_proj_lin(s, *_):
    return mindep.randlin_projective(s)[-1]

def rand_proj_lin_opt(**kwds):
    @sampler
    def rs(s, *_):
        return mindep.randlin_projective(s, **kwds)[-1]
    return rs

@sampler
def rand_weighted(s, *_): # redo these using WeightedLin class
    return opt_mindep.randlin_fixed_weights(s, thing_fn=CONDITIONING, head_final=False)[-1]

@rfutils.memoize
def get_weights(lang, i, deptypes, head_final):
    weights = opt_mindep.rand_fixed_weights(deptypes, head_final=head_final)
    return weights

@sampler
def rand_weighted_per_lang(s, lang, i, deptypes):
    weights = get_weights(lang, i, deptypes, False)
    return opt_mindep.randlin_from_weights(s, weights, CONDITIONING)[-1]

@sampler
def rand_weighted_headfinal_per_lang(s, lang, i, deptypes):
    weights = get_weights(lang, i, deptypes, True)
    return opt_mindep.randlin_from_weights(s, weights, opt_mindep.get_deptype)[-1]

@sampler
def rand_weighted_headfinal(s, *_):
    return opt_mindep.randlin_fixed_weights(s, thing_fn=opt_mindep.get_deptype, head_final=True)[-1]

@sampler
def rand_fullyfree(s, *_):
    lin = [n for n in s.nodes() if n != 0]
    random.shuffle(lin)
    lin.insert(0, 0)
    return lin

# Model-based random lin functions

def rand_proj_lin_spec(spec):
    @sampler
    def rand_proj_lin(s, lang, *_):
        m = load_linearization_model(lang, spec)
        return lin.proj_lin(m, s)
    return random_sample_proj_lin


# depvar functions will be mindep.deplen, mindep.deplens, mindep.deptypes, etc.

def build_it(sentences, lang, parallel=False):
    return generate_rows(
        sentences,
        lang,
        # linearization functions
        {            
            'real': identity_lin,
            #'rand_proj_lin': rand_proj_lin,
            'rand': rand_fullyfree,
        },
        # dependent variables
        { 
            'deplen': mindep.deplens,
        },
        # background variables
        {
            'deptype': deptypes,
            'dpos': get_dep_attrs(cond.get_pos),
            'hpos': get_head_attrs(cond.get_pos),
            'dword': get_dep_attrs(cond.get_word),
            'hword': get_head_attrs(cond.get_word),
        },
        parallel=parallel,
    )
    
def postprocess(df):
    import pandas as pd
    dfm = pd.melt(df, id_vars='lang length start_line'.split())
    dfm['real'] = dfm['variable'].map(name_fn)
    del dfm['variable']
    return dfm

#executor = Executor()

def pmap(f, xs):
    for future in as_completed(executor.map(f, xs)):
        yield future.result()

def main(cmd, *args):
    if cmd == "run":
        langs = tuple(args)
        if langs == ('ud',):
            langs = corpora.ud_langs
        rows = rfutils.flat(
            build_it(
                corpora.ud_corpora[lang].sentences(**OPTS),
                lang,
                parallel=False
            )
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
        df = functools.reduce(pd.DataFrame.append, map(pd.read_csv, filenames))
        new_df = postprocess(df)
        new_df.to_csv(sys.stdout)
    else:
        rfutils.err("Unknown command: %s" % cmd)
        
if __name__ == '__main__':
    main(*sys.argv[1:])
    

#!/usr/bin/python3
""" Interfaces to corpora. """
from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import
from .compat import *

import os
import glob
import path
import socket

import networkx as nx

from .readcorpora import *

try:
    username = os.getlogin()
except OSError:
    username = None

# Replace with your username and local directory to make things speedier    
if username == 'canjo':
    data_dir = "/Users/canjo/data/cliqs/"
else:
    data_dir = "http://tedlab.mit.edu/datasets/cliqs/"


ud_path = "ud-treebanks-v2.5/"
sud_path = "sud-treebanks-v2.4_2019_08_13/"
udt_path_template = data_dir + "universal_treebanks_v1.0/%s/%s-universal.conll"
udt2_path_template = data_dir + "utb2_std/%s/all.conll"
hamledt_stanford_path_template = data_dir + "hamledt2/%s/stanford/conll/all.conll"
hamledt2_stanford_path_template = data_dir + "2.0/%s/stanford/all.conll"
hamledt2plus_stanford_path_template = data_dir + "hamledt2plus/%s/stanford/all.conll"
hamledt3_stanford_path_template = data_dir + "hamledt3/%s/all.conllu" 
ud_path_template = data_dir + ud_path + "%s/all.conllu"
sud_path_template = data_dir + sud_path + "%s/all.conllu"
ud_train_template = data_dir + ud_path + "%s/%s-ud-train.conllu"
ud_dev_template = data_dir + ud_path + "%s/%s-ud-dev.conllu"
ud_test_template = data_dir + ud_path + "%s/%s-ud-test.conllu"
proiel_torot_template = data_dir + "proiel/%s/all.conll"
proiel_template = data_dir + "proiel/proiel-treebank-20150725/%s.conll"
torot_template = data_dir + "proiel/torot/%s.conll"
mlc_template = data_dir + "MLC_data/UD_%s/%s.conllu"
gd_path_template = "/Users/canjo/code/cliqs/gdoutput/%s-%d.conllu"

# Now let's keep track of all our data in dictionaries.

udt_corpora = {    
    "es" : UDTDependencyTreebank(udt_path_template % ("es", "es")),
    "fr" : UDTDependencyTreebank(udt_path_template % ("fr", "fr")),
    "ko" : UDTDependencyTreebank(udt_path_template % ("ko", "ko")),
    "id" : UDTDependencyTreebank(udt2_path_template % "id"),
    "it" : UDTDependencyTreebank(udt2_path_template % "it"),
    "de" : UDTDependencyTreebank(udt2_path_template % "de"),           
    "ja" : UDTDependencyTreebank(udt2_path_template % "ja"), 
    "pt-br" : UDTDependencyTreebank(udt2_path_template % "pt-br"),
    "sv" : UDTDependencyTreebank(udt2_path_template % "sv"),
}

ud_langs = "af aii akk am ar be bg bho bm br bxr ca cop cs cu cy da de el en es et eu fa fi fo fr fro ga gd gl got grc gsw gun he hi hr hsb hu hy id it ja kk kmr ko koi kpv krl la lt lv lzh mdf mr mt myv nl no olo orv pcm pl pt qhe ro ru sa sk sl sme sms sr sv swl ta te th tl tr ug uk ur vi wbp wo yo yue zh".split()

ud_corpus_names = [
    str(path.Path(x).basename()) for x in glob.glob(data_dir + ud_path + "UD_*")
]

sud_corpus_names = [
    str(path.Path(x).basename()) for x in glob.glob(data_dir + sud_path + "SUD_*")
]

pud_langs = "ar_pud cs_pud de_pud en_pud es_pud fi_pud fr_pud hi_pud it_pud ja_pud pt_pud ru_pud sv_pud tr_pud zh_pud".split()

ud_corpora = { 
    lang : UniversalDependency1Treebank(ud_path_template % lang)
    for lang in ud_langs
}

all_ud_corpora = {
    lang : UniversalDependency1Treebank(ud_path_template % lang)
    for lang in ud_corpus_names
}

sud_langs = "af aii akk am ar be bg bm br bxr ca cop cs cu cy da de el en es et eu fa fi fo fr fro ga gl got grc gun he hi hr hsb hu hy id it ja kk kmr ko kpv krl la lt lv lzh mr mt myv nl no orv pcm pl pt qhe ro ru sa sk sl sme sr sv swl ta te th tl tr ug uk ur vi wbp wo yo yue zh".split()

sud_corpora = {
    lang : UniversalDependency1Treebank(sud_path_template % lang)
    for lang in sud_langs
}
    
gd_corpora = [
    {
        lang : UniversalDependency1Treebank(gd_path_template % (lang, i))
        for lang in ud_langs
    }
    for i in range(24)
]
    

parallel_corpora = {
    lang : UniversalDependency1Treebank(ud_path_template % lang)
    for lang in pud_langs
}

def corpus_path(directory, lang, which):
    matches = glob.glob(data_dir + ud_path + "%s/*-ud-%s.conllu" % (lang, which))
    if len(matches) == 1:
        return matches[0]
    else:
        return None

ud_train_corpora = {
    lang : UniversalDependency1Treebank(corpus_path(data_dir + ud_path, lang, 'train'))
    for lang in ud_corpora
    if corpus_path(data_dir + ud_path, lang, 'train') is not None
}

ud_dev_corpora = {
    lang : UniversalDependency1Treebank(corpus_path(data_dir + ud_path, lang, 'dev'))    
    for lang in ud_corpora
    if corpus_path(data_dir + ud_path, lang, 'dev') is not None    
}

ud_test_corpora = {
    lang : UniversalDependency1Treebank(corpus_path(data_dir + ud_path, lang, 'test'))
    for lang in ud_langs
    if corpus_path(data_dir + ud_path, lang, 'test') is not None    
}


all_ud_train_corpora = {
    lang : UniversalDependency1Treebank(corpus_path(data_dir + ud_path, lang, 'train'))
    for lang in all_ud_corpora
    if corpus_path(data_dir + ud_path, lang, 'train') is not None
}

all_ud_dev_corpora = {
    lang : UniversalDependency1Treebank(corpus_path(data_dir + ud_path, lang, 'dev'))    
    for lang in all_ud_corpora
    if corpus_path(data_dir + ud_path, lang, 'dev') is not None    
}

all_ud_test_corpora = {
    lang : UniversalDependency1Treebank(corpus_path(data_dir + ud_path, lang, 'test'))
    for lang in all_ud_corpora
    if corpus_path(data_dir + ud_path, lang, 'test') is not None    
}
    
proiel_torot_langs = "orv cu xcl got grc la".split()

proiel_torot_corpora = {
    lang : CoNLLDependencyTreebank(proiel_torot_template % lang)
    for lang in proiel_torot_langs
}

mlc_langs = "Afrikaans Arabic Basque Bulgarian Catalan Chinese Croatian Czech Danish Dutch English Estonian Finnish French Galician Greek Hebrew Hindi Hungarian Italian Latvian Norwegian-Bokmaal Norwegian-Nynorsk Persian Polish Portuguese Romanian Russian-SynTagRus Serbian Slovak Slovenian Spanish-AnCora Swedish Turkish Ukrainian Urdu Vietnamese".split()
assert len(mlc_langs) == 37


mlc_corpora = {
    lang : UniversalDependency1Treebank(mlc_template % (lang, lang))
    for lang in mlc_langs
}

nt_corpora = {
    'grc': CoNLLDependencyTreebank(proiel_template % 'greek-nt'),
    'la': CoNLLDependencyTreebank(proiel_template % 'latin-nt'),
    'got': CoNLLDependencyTreebank(proiel_template % 'gothic-nt'),
    'xcl': CoNLLDependencyTreebank(proiel_template % 'armenian-nt'),
    'cu': CoNLLDependencyTreebank(proiel_template % 'marianus'),
}

hamledt2_langs = "ta de".split()
hamledt2_corpora = {
    lang : CoNLLDependencyTreebank(hamledt2_stanford_path_template % lang)
    for lang in hamledt2_langs
}

hamledt3_langs = "ja it de es fi ar ta ro nl fa da cs la grc pt sl bg hu sv sl he tr en et bn ca el eu sk te hi hr".split()
hamledt_corpora = {
    lang : UniversalDependency1Treebank(hamledt3_stanford_path_template % lang)
    for lang in hamledt3_langs
}
hamledt_corpora['ru'] = CoNLLDependencyTreebank(hamledt2plus_stanford_path_template % "ru")

# Policy: Use UD if available, else HamleDT, else UDT, else PROIEL-TOROT, else whatever
corpora = {}
corpora.update(proiel_torot_corpora) # xcl, orv
corpora.update(udt_corpora) # ko
corpora.update(hamledt_corpora) # bn, ca, ru, sk, te, tr
corpora.update(ud_corpora)
corpora.update(parallel_corpora)
corpora.update({'ru_syntagrus' : UniversalDependency1Treebank(ud_path_template % 'ru_syntagrus')})

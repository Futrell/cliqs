#!/usr/bin/python3
""" Interfaces to corpora. """
from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import
from .compat import *

import os
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

udt_path_template = data_dir + "universal_treebanks_v1.0/%s/%s-universal.conll"
udt2_path_template = data_dir + "utb2_std/%s/all.conll"
hamledt_stanford_path_template = data_dir + "hamledt2/%s/stanford/conll/all.conll"
hamledt2_stanford_path_template = data_dir + "2.0/%s/stanford/all.conll"
hamledt2plus_stanford_path_template = data_dir + "hamledt2plus/%s/stanford/all.conll"
hamledt3_stanford_path_template = data_dir + "hamledt3/%s/all.conllu"
ud_path_template = data_dir + "ud-treebanks-v2.0/%s/all.conllu"
ud_train_template = data_dir + "ud-treebanks-v2.0/%s/%s-ud-train.conllu"
ud_dev_template = data_dir + "ud-treebanks-v2.0/%s/%s-ud-dev.conllu"
ud_test_template = data_dir + "ud-treebanks-v2.0/%s/%s-ud-test.conllu"
proiel_torot_template = data_dir + "proiel/%s/all.conll"
proiel_template = data_dir + "proiel/proiel-treebank-20150725/%s.conll"
torot_template = data_dir + "proiel/torot/%s.conll"    


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

ud_langs = "grc ar eu bg ca zh hr cs da nl en et fi fr gl de got grc he hi hu id ga it ja kk la lv no cu fa pl pt ro ru sl es sv ta tr el swl uk ug vi sk sa cop".split()

ud_corpora = {
    lang : UniversalDependency1Treebank(ud_path_template % lang)
    for lang in ud_langs
}

ud_train_corpora = {
    lang : UniversalDependency1Treebank(ud_train_template % (lang, lang))
    for lang in ud_corpora
}

ud_dev_corpora = {
    lang : UniversalDependency1Treebank(ud_dev_template % (lang, lang))
    for lang in ud_corpora
}

ud_test_corpora = {
    lang : UniversalDependency1Treebank(ud_test_template % (lang, lang))
    for lang in ud_corpora
}

proiel_torot_langs = "orv cu xcl got grc la".split()

proiel_torot_corpora = {
    lang : CoNLLDependencyTreebank(proiel_torot_template % lang)
    for lang in proiel_torot_langs
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

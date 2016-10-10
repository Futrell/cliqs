#!/usr/bin/python3
from __future__ import print_function
import io
import re
import gzip
import copy
import codecs
import functools
from collections import namedtuple

import pyrsistent as pyr
import networkx as nx
import requests

import deptransform
import depgraph

EMPTY_SET = frozenset({})
CH_CONVERSION_ORDER = ['case', 'cop', 'mark']

def myopen(filename, **kwds):
    if filename.startswith("http"):
        assert not filename.endswith(".gz")
        return io.StringIO(requests.get(filename).text)
    elif filename.endswith('.gz'):
        open_file = gzip.open(filename, mode='rb', **kwds)
    else:
        open_file = open(filename, mode='rb', **kwds)
    return codecs.getreader('utf-8')(open_file)

class DepSentence(nx.DiGraph):
    """ Dependency Sentence

    Directed graph representation of the dependency parse of a sentence.
    Contains extra information on data sources.

    """
    def __init__(self, filename=None, start_line=None, end_line=None, ch=None, high=None):
        self.start_line = start_line
        self.end_line = end_line
        self.filename = filename
        self.ch = ch
        self.high = high
        super(DepSentence, self).__init__()

    def __repr__(self):
        argstr = ", ".join(map(str, filter(None, [self.start_line, self.end_line])))
        return "DepSentence('{}', {})".format(str(self.filename), self.start_line)

    __str__ = __repr__

    @classmethod
    def from_digraph(cls, digraph):
        self = cls()
        self.add_nodes_from(digraph.nodes(data=True))
        self.add_edges_from(digraph.edges(data=True))
        return self

    def add_word(self, word_id, word_attr, head_id, rel_attr):
        self.add_node(word_id)
        self.node[word_id].update(word_attr)
        self.node[word_id]['id'] = word_id
        self.add_edge(head_id, word_id, deptype=rel_attr)

# from_content_head : graph x [string] -> Maybe graph        
def from_content_head(ds, rels, verbose=False, strict=False):
    for rel in rels:
        ds = from_content_head_rel(ds, rel, verbose=verbose, strict=strict)
        if ds is None: 
            return ds
    return ds

# from_content_head_rel : graph x string -> Maybe graph
def from_content_head_rel(ds, rel, verbose=False, strict=False):
    if rel not in ds.ch or not ds.ch[rel]:
        return ds
    else:
        new_ds = deptransform.reverse_content_head(
            ds,
            rel,
            ds.high[rel],
            verbose=verbose,
            strict=strict,
        )
        if new_ds is None:
            return new_ds
        else:
            new_ds.ch = ds.ch.set(rel, False)
            new_ds.high = ds.high.set(rel, EMPTY_SET)
            return new_ds

class DependencyTreebank(object):
    """ Interface to Dependency Treebanks. """
    ch = pyr.m()
    high = pyr.m()
    
    def __init__(self, filename, load_into_memory=False, ch={}, high={}):
        self.filename = filename
        self._sentences = []
        self._sentences_in_memory_flags = {}

        self.ch = self.ch.update(ch)
        self.high = self.high.update(high)

    def __repr__(self):
        # TODO update this to show properties
        DT = type(self).__name__
        return """%s("%s")""" % (DT, self.filename)

    __str__ = __repr__

    def load_into_memory(self, verbose=True, **kwds):
        self._sentences = []
        self._sentences = list(self.sentences(verbose=True, **kwds))
        self._sentences_in_memory_flags = kwds

    def read(self):
        return myopen(self.filename)

    def sentences(self, verbose=False,
                  strict=False,
                  remove_punct=True,
                  fix_content_head=CH_CONVERSION_ORDER,
                  allow_multihead=False,
                  allow_multiple_roots=False):
        """ Yield sentences as DepSentences. """
        def gen():
            with self.read() as lines:
                for sentence in self.generate_sentences(
                        lines,
                        allow_multihead=allow_multihead,
                        allow_multiple_roots=allow_multiple_roots,
                        verbose=verbose):
                    sentence.ch = self.ch
                    sentence.high = self.high
                    if remove_punct:
                        sentence = deptransform.remove_punct_from_sentence(
                            sentence,
                            verbose=verbose,
                        )
                        if sentence is None:
                            continue
                    if fix_content_head:
                        sentence = from_content_head(
                            sentence,
                            fix_content_head,
                            verbose=verbose,
                            strict=strict
                        )
                        if sentence is None:
                            continue
                    if not allow_multihead and len(sentence.out_edges(0)) > 1:
                        continue
                    yield sentence
        if self._sentences:
            return self._sentences
        else:
            return gen()


# This class and the following mostly provide parsers for the various formats.

class CoNLLDependencyTreebank(DependencyTreebank):
    """ A dependency treebank in CoNLL format. """
    word_id_col = 0
    word_col = 1
    lemma_col = 2
    pos_col = 3
    pos2_col = 4
    infl_col = 5
    head_id_col = 6
    deptype_col = 7

    def analyze_line(self, parts, verbose=False):
        """ Analyze a line of the CoNLL formatted file giving word id, word_attr
         dict, head id, and dep type.
         """
        # lots of accumulated ad-hoc fixes in here 
        try:
            try:
                word_id = int(parts[self.word_id_col])
            except ValueError:
                word_id = int(float(parts[self.word_id_col])) 
        
            word_attr = {
                'word' : parts[self.word_col],
                'lemma' : parts[self.lemma_col].split("+")[0],
                'pos' : parts[self.pos_col],
                'pos2' : parts[self.pos2_col],
                'infl' : parts[self.infl_col],
            }
        
            try:
                head_id = int(parts[self.head_id_col])
            except ValueError:
                head_id = int(float(parts[self.head_id_col])) 
            deptype = parts[self.deptype_col]

            if word_attr['pos'] == "_" or word_attr['pos'] == "X":
                if "=" not in word_attr['infl']:
                    infls = {}
                else:
                    infls = dict(kv.split("=") for kv in word_attr['infl'].split("|"))
                    # try to recover some POS information from the infl field
                    if 'prontype' in infls:
                        word_attr['pos'] = 'PRON'
                    elif 'subpos' in infls and infls['subpos'] == 'det':
                        word_attr['pos'] = 'DET'
                    elif 'pos' in infls:
                        word_attr['pos'] = infls['pos'].upper()
            
            return word_id, word_attr, head_id, deptype
        except Exception as e:
            if verbose:
                print(
                    "Parsing error in file: %s %s" % (self.filename, e),
                    file=sys.stderr
                )
                print("Offending line: %s" % parts, file=sys.stderr)
            return None

    def sentence_lines(self, lines):
        """ [CoNLL line] -> [([CoNLL line], Int, Int)] """
        start_line = 0
        sentence_lines = []
        for i, line in enumerate(lines, 1):
            line = line.strip()
            if line:
                sentence_lines.append(line)
            else:
                if sentence_lines:
                    end_line = i
                    yield sentence_lines, start_line, end_line
                sentence_lines = []
                start_line = i + 1
        if sentence_lines:
            end_line = i
            yield sentence_lines, start_line, end_line

    def analyze_sentence_lines(self,
                               lines,
                               allow_multihead=False,
                               verbose=False):
        sentence = DepSentence(self.filename)
        for line in lines:
            if not line.startswith("#"):
                parts = line.split("\t")
                word_parts = self.analyze_line(parts, verbose=verbose)
                if word_parts:
                    sentence.add_word(*word_parts)
        return sentence
    
    def generate_sentences(self,
                           lines,
                           allow_multihead=False,
                           allow_multiple_roots=False,
                           verbose=False):
        """ Segment an iterable of CoNLL lines into sentences and analyze them.
        [CoNLL line] -> [DepSentence]
        """
        for sentence_lines, i_start, i_end in self.sentence_lines(lines):
            sentence = self.analyze_sentence_lines(sentence_lines,
                                                allow_multihead=allow_multihead,
                                                verbose=verbose)
            if not sentence.nodes():
                continue # drop empty sentences
            if not allow_multiple_roots:
                if not depgraph.is_singly_rooted(sentence):
                    continue # drop non singly rooted trees
            sentence.start_line = i_start
            sentence.end_line = i_end
            sentence.filename = self.filename
            yield sentence

class DundeeTreebank(CoNLLDependencyTreebank):
    word_id_col = 3
    word_col = 0 # dummy
    lemma_col = 0 # dummy
    pos_col = 4
    pos2_col = 4 # dummy
    infl_col = 0 # dummy
    head_id_col = 5
    deptype_col = 6

    def generate_sentences(self, lines, **ignore):
        prev_sentid = None
        curr_sentence = None
        for line in lines:
            itemno, wnum, sentid, id, cpos, head, deprel = line.split("\t")
            if sentid != prev_sentid:
                if curr_sentence is not None: # not first time through
                    yield curr_sentence 
                curr_sentence = DepSentence(self.filename)
            prev_sentid = sentid
            curr_sentence.add_word(int(id),
                                   {'pos': cpos},
                                   int(head),
                                   deprel)
        yield curr_sentence

    def read(self):
        lines = myopen(self.filename)
        next(lines) # throw away the header line
        return lines
            

class UDTDependencyTreebank(CoNLLDependencyTreebank):
    pass

# based on but NOT IDENTICAL TO https://universaldependencies.github.io/docs/u/dep/index.html
# I've customized the categories to my own purposes here.
UD_CLAUSAL_CORE_RELS = set("nsubj nsubjpass csubj csubjpass dobj iobj ccomp xcomp".split())
UD_CLAUSAL_NONCORE_RELS = set("nmod advcl advmod neg expl".split())
UD_NOUN_RELS = set("nummod appos nmod acl amod det neg".split())
UD_MARKING_RELS = set("case mark".split())
UD_COMPOUNDING_RELS = set("compound name mwe foreign goeswith".split())
UD_JOINING_RELS = set("list dislocated parataxis remnant reparandum".split())
UD_WEIRD_VERB_RELS = set("aux auxpass cop".split())
UD_DISCOURSE_RELS = set("vocative discourse".split())
UD_COORDINATION_RELS = set("conj cc".split())

class UniversalDependency1Treebank(CoNLLDependencyTreebank):
    ch = pyr.pmap({
        'mark': True,
        'case': True,
        'cop': True,
        'aux': True,
    })

    high = pyr.pmap({
        'case': (
            UD_CLAUSAL_CORE_RELS
            | UD_CLAUSAL_NONCORE_RELS - {'nmod'}
            | UD_WEIRD_VERB_RELS
            | {'mark'}
        ),
        'cop': (
            UD_CLAUSAL_CORE_RELS - {'dobj'} # copulas never have dobj???
            | UD_CLAUSAL_NONCORE_RELS
            | UD_JOINING_RELS
            | UD_WEIRD_VERB_RELS
            | UD_DISCOURSE_RELS
            | UD_COORDINATION_RELS # "but" always attaches high, "and" often attaches low...
            | {'mark', 'nmod/tmod'}
        ),
        'mark': EMPTY_SET,
        'aux': EMPTY_SET,
    })
    

    def analyze_compound_line(self, parts, verbose=False):
        id_lower, id_upper = map(int, parts[0].split("-"))
        word_ids = tuple(range(id_lower, id_upper+1))

        form = parts[1]
        info = {
            'part_of': word_ids,
            'form': form,
        }
        return word_ids, info

    def analyze_conllu_misc(self, parts, verbose=False):
        return parts[-1].strip("_")

    def analyze_conllu_extra_heads(self, parts, verbose=False):
        if parts[-2] == "_":
            return []
        else:
            extra_heads = parts[-2].split("|")
            extra_heads = [dep.split(":", 1) for dep in extra_heads]
            return [(int(h_id), deptype) for h_id, deptype in extra_heads]

    def analyze_sentence_lines(self,
                               lines,
                               allow_multihead=False,
                               verbose=False):
        lines = list(lines)
        sentence = DepSentence(filename=self.filename)
        for line in lines:
            if not line.startswith("#"):
                parts = line.split("\t")
                if "-" in parts[0]:
                    word_ids, info = self.analyze_compound_line(parts,
                                                                verbose=verbose)
                    for word_id in word_ids:
                        sentence.add_node(word_id)
                        sentence.node[word_id].update(info)
                else:
                    # first analyze line as if it was CoNLL
                    word_parts = self.analyze_line(parts, verbose=verbose)
                    if not word_parts:
                        continue
                    sentence.add_word(*word_parts)

                    # then do extra stuff for CoNLL-U
                    word_id = word_parts[0]
                    word = sentence.node[word_id]
                    if 'form' not in word:
                        word = sentence.node[word_id]
                        word['form'] = word['word']
                        word['part_of'] = (word_id,)

                    word['misc'] = self.analyze_conllu_misc(parts,
                                                            verbose=verbose)
                    if allow_multihead:
                        extra_heads = self.analyze_conllu_extra_heads(parts, 
                                                                verbose=verbose)
                        for head_id, deptype in extra_heads:
                            sentence.add_edge(head_id,
                                              word_id,
                                              deptype=deptype)
        return sentence


class TurkuDependencyTreebank(CoNLLDependencyTreebank):
    word_id_col = 0
    word_col = 1
    lemma_col = 2
    pos_col = 4
    pos2_col = 5
    infl_col = 6
    head_id_col = 7
    deptype_col = 8


# For Treebanks in Perseus XML format: Greek and Latin

class PerseusDependencyTreebank(DependencyTreebank):
    """ A dependency treebank in Perseus format """
    # let's parse XML with regexes!!!!!!!!!!!! hoorah!!!!!
    id_re = re.compile(" id=\"([^\"]*)\" ")
    form_re = re.compile(" form=\"([^\"]*)\" ")
    lemma_re = re.compile(" lemma=\"([^\"]*)\" ")
    pos_re = re.compile(" postag=\"([^\"]*)\" ")
    head_id_re = re.compile(" head=\"([^\"]*)\" ")
    deptype_re = re.compile(" relation=\"([^\"]*)\" ")

    def analyze_line(self, line, verbose=False):
        line = line.strip()
        word_id = int(self.id_re.findall(line)[0])
        
        word_attr = {}
        try:
            word_attr['word'] = self.form_re.findall(line)[0]
        except IndexError:
            print("No word found in: %s" % line)
        
        try:
            word_attr['lemma'] = self.lemma_re.findall(line)[0]
        except IndexError:
            print("No lemma found in: %s" % line)
        
        try:
             word_attr['pos'] = self.pos_re.findall(line)[0]
        except IndexError:
            print("No pos tag found in: %s" % line)
        
        head_id = int(self.head_id_re.findall(line)[0])
        deptype = self.deptype_re.findall(line)[0]

        return word_id, word_attr, head_id, deptype

    def generate_sentences(self,
                           lines,
                           allow_multihead=False,
                           allow_multiple_roots=False,
                           verbose=False):
        sentence_so_far = DepSentence(filename=self.filename)
        for line in lines:
            line = line.strip()
            if line and line.startswith("<word"):
                word_parts = self.analyze_line(line,
                                               allow_multihead=allow_multihead,
                                               erbose=verbose)
                sentence_so_far.add_word(*word_parts)
            else:
                if line.startswith("<sentence"):
                    if sentence_so_far.nodes():
                        yield sentence_so_far
                        sentence_so_far = DepSentence(self.filename)
                # otherwise do nothing
        if sentence_so_far.nodes():
            if allow_multiple_roots or depgraph.is_singly_rooted(sentence_so_far):
                yield sentence_so_far


class StanfordDependencyTreebank(DependencyTreebank):
    """ Stanford Dependency Treebank: Not necessarily trees! """
    def generate_sentences(self,
                           lines,
                           allow_multihead=False,
                           allow_multiple_roots=False,
                           verbose=False):
        if allow_multihead:
            raise NotImplementedError
        sentence_so_far = DepSentence(filename=self.filename)
        for line in lines:
            line = line.strip()
            if line:
                relation, words = line.split("(", 1)
                part1, part2 = words.split(", ")
                head_id, head_attr = self.analyze_word(part1) # kill leading (
                dep_id, dep_attr = self.analyze_word(part2[:-1]) # trailing )
                sentence_so_far.add_node(head_id, head_attr)
                sentence_so_far.add_word(dep_id, dep_attr, head_id, relation)
            else:
                yield sentence_so_far
                sentence_so_far = DepSentence(filename=self.filename)
        if sentence_so_far.nodes():
            if allow_multiple_roots or depgraph.is_singly_rooted(sentence_so_far):
                yield sentence_so_far            

    def analyze_word(self, stuff):
        attributes = {}
        stuff, word_id = stuff.rsplit("-", 1)
        attributes['word'] = stuff
        if "'" in word_id:
            word_id = word_id.replace("'", "")
        return int(word_id), attributes

# And here's the parser for the Parsed Gigaword corpus.

class ParsedGigawordDependencyTreebank(StanfordDependencyTreebank):
    """ parsed gigaword dependency treebank

    sentences() yields DepSentence objects out of a file with lines
    like in the parsed gigaword that Sam and I made:
    i.e. lines like det(Vikings|Vikings^NNPS-2, The|the^DT-1)
    with empty lines indicating sentence breaks.

    """
    def analyze_word(self, stuff):
        attributes = {}
        stuff, word_id = stuff.rsplit("-", 1)
        try:
            attributes["word"], stuff = stuff.split("|", 1)
            attributes["lemma"], stuff = stuff.split("^", 1)
            attributes["pos"] = stuff
        except ValueError: # it'll fail for ROOT-0
            pass
        if "'" in word_id: # happens when the graph is weird
            word_id = word_id.replace("'", "") 
        return int(word_id), attributes


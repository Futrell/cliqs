import sys
import csv
from collections import Counter

import cliqs.corpora as corpora
import cliqs.conditioning as cond
import cliqs.entropy as entropy

SENTENCE_OPTS = {'fix_content_head': False, 'collapse_flat': True}
CONDITIONING = cond.get_pos

def hdpmi(f, sentences):
    def gen():
        for s in sentences:
            for h, d in s.edges():
                if h != 0:
                    head = f(s, h)
                    dep = f(s, d)
                    yield head, dep
    counts = Counter(gen())
    return entropy.pointwise_mutual_informations(counts)

def main(langs):
    writer = csv.writer(sys.stdout)
    writer.writerow(['lang', 'h', 'd', 'pmi'])
    if langs == ['all']:
        langs = corpora.ud_corpora.keys()
    for lang in langs:
        sentences = corpora.ud_corpora[lang].sentences(**SENTENCE_OPTS)
        for (h, d), pmi in hdpmi(CONDITIONING, sentences):
            writer.writerow([lang, h, d, pmi])
            

if __name__ == '__main__':
    main(sys.argv[1:])
    
    

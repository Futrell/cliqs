# Crosslinguistic Investigations in Quantitative Syntax (CLIQS)

[![Build Status](https://travis-ci.org/Futrell/cliqs.svg?branch=master)](https://travis-ci.org/Futrell/cliqs)

This is code for studying quantitative syntax using dependency corpora.

It is written for Python 3.5, and has been tested to work in Python 2.7.

### Dependencies

`pip install -r requirements.txt` for basic functionality.
Additionally, `pip install -r optrequirements.txt` for optional dependencies used for parallelization and visualization.

### Example

The list of langs can be found at `corpora.ud_langs`.
To compare dependency length in some languages to random and minimal baselines, run:
`python run_mindep.py run lang1 lang2 ... langn > result_raw.csv`.

Then postprocess the resulting csv:
`python run_mindep.py postprocess result_raw.csv > result.csv`.



# Crosslinguistic Investigations in Quantitative Syntax (CLIQS)

This is code for studying quantitative syntax using dependency corpora.

It is written for Python 3.5, and seems to work in Python 2.7 also, but I can't guarantee it.

### Dependencies

`pip install -r requirements.txt` for basic functionality.
Additionally, `pip install -r optrequirements.txt` for optional dependencies used for parallelization and visualization.

## Building

`python run_mindep.py run lang1 lang2 ... langn` for an initial run.
The list of langs can be found at `corpora.ud_langs`.

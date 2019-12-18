# Crosslinguistic Investigations in Quantitative Syntax (CLIQS)

This is code for studying quantitative syntax using dependency corpora.

It is written for Python 3.5, and has been tested to work in Python 2.7.

### Dependencies

`pip install -r requirements.txt` for basic functionality.
Additionally, `pip install -r optrequirements.txt` for optional dependencies used for parallelization and visualization.

### Setting up corpora
Supposing you have your Universal Dependencies treebanks, as downloaded from the UD website, in a directory `$UD_DIR`. Copy the file `process_ud.sh` into `$UD_DIR` and run it with `sh process_ud.sh`. This will rename the directories into the format that `cliqs` expects. Then modify the path in `cliqs/corpora.py` to reflect your UD path.


### Example

The list of langs can be found at `corpora.ud_langs`.

To compare dependency length in some languages to random and minimal baselines, run:
`python run_mindep.py run lang1 lang2 ... langn > result_raw.csv`.

Then postprocess the resulting csv:
`python run_mindep.py postprocess result_raw.csv > result.csv`.

Then you can run the various R scripts starting in `mindep_` to analyze the results and generate figures.

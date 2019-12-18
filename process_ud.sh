#!/usr/bin/env bash
# Make concatenated files; make shortcodes with iso codes

for lang in `ls . | grep UD_`; do
    echo $lang
    pushd $lang
        cat *-ud-*.conllu > all.conllu;
        code=`ls | grep "ud-test.conllu" | sed "s/-.*//g"`
        popd
    ln -s $lang $code;
done

# For each language, take the largest corpus to be its representative

CHOSEN_CORPORA=$(wc -l */all.conllu | grep -v UD | grep -v total | sed "s/\/all.conllu//g" | sed "s/_/ /g" | sort -r -nk1,1 | sort -sk 2,2 | awk -F" " '{if (l != $2) { print $2"_"$3; l=$2 }}')

for corpus in $CHOSEN_CORPORA; do
    lang=$(awk -F"_" '{print $1}' <<< $corpus)
    ln -s $corpus $lang;
done

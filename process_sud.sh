# Make concatenated files; make shortcodes with iso codes

for lang in `ls . | grep SUD_`; do
    echo $lang
    pushd $lang
        cat *-sud-*.conllu > all.conllu;
        code=`ls | grep "sud-test.conllu" | sed "s/-.*//g"`
        popd
    ln -s $lang $code;
done

# For each language, take the largest corpus to be its representative

CHOSEN_CORPORA=$(wc -l */all.conllu | grep -v SUD | grep -v total | sed "s/\/all.conllu//g" | sed "s/_/ /g" | sort -r -k2,2 -k1,1 | awk -F" " '{if (l != $2) { print $2"_"$3; l=$2 }}' | sed '1d')

for corpus in $CHOSEN_CORPORA; do
    lang=$(awk -F"_" '{print $1}' <<< $corpus)
    ln -s $corpus $lang;
done


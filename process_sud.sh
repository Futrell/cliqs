# Make concatenated files; make shortcodes with iso codes

for lang in `ls . | grep SUD_`; do
    pushd $lang;
        cat *-sud-*.conllu > all.conllu;
        code=`ls | grep "sud-test.conllu" | sed "s/-.*//g"`
        popd;
    ln -s $lang $code;
done

# For each language, take the largest corpus to be its representative

CHOSEN_CORPORA=$(wc -l */all.conllu | # get line counts
    grep -v SUD | # look at each corpus only once
    grep -v total | # irrelevant line
    sed "s/\/all.conllu//g" | # peel off filename
    sed "s/_/ /g" | # replace _ with space so now we have three columns
    sort -r -k2,2 -k1,1 | # reverse-sort by language and then by count
    awk -F" " '{if (l != $2) { print l"_"c; l=$2; c=$3 }}' |
    sed '1d')

for corpus in $CHOSEN_CORPORA; do
    lang=$(awk -F"_" '{print $1}' <<< $corpus)
    ln -s $corpus $lang;
done


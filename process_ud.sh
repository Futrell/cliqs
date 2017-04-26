# Make concatenated files; make shortcodes with iso codes

for lang in `ls . | grep UD`; do
    cd $lang;
    cat *-ud-*.conllu > all.conllu;
    code=`ls | grep "ud-dev.conllu" | sed "s/-.*//g"`;
    cd ..;
    ln -s $lang $code;
done

# Choose which corpora will be the "main" one for languages.
# Override links from above.

ln -s UD_Arabic-NYUAD ar
ln -s UD_Arabic ar_ud

ln -s UD_Ancient_Greek-PROIEL grc
ln -s UD_Ancient_Greek grc_perseus

ln -s UD_Latin-PROIEL la
ln -s UD_Latin la_perseus

ln -s UD_Russian-SynTagRus ru
ln -s UD_Russian ru_ud

ln -s UD_Spanish-AnCora es
ln -s UD_Spanish es_ud


for lang in `ls . | grep UD`; do
    pushd $lang;
    cat *-ud-*.conllu > all.conllu;
    popd;
done

ln -s UD_Ancient_Greek-PROIEL grc
ln -s UD_Arabic ar
ln -s UD_Baseque eu
ln -s UD_Bulgarian bg
ln -s UD_Catalan ca
ln -s UD_Chinese zh
ln -s UD_Croatian hr
ln -s UD_Czech cs
ln -s UD_Danish da
ln -s UD_Dutch nl
ln -s UD_English en
ln -s UD_Estonian et
ln -s UD_Finnish fi
ln -s UD_French fr
ln -s UD_Galician gl
ln -s UD_German de
ln -s UD_Gothic got
ln -s UD_Greek el
ln -s UD_Hebrew he
ln -s UD_Hindi hi
ln -s UD_Hungarian hu
ln -s UD_Indonesian id
ln -s UD_Irish ga
ln -s UD_Italian it
ln -s UD_Japanese-KTC ja
ln -s UD_Kazakh kk
ln -s UD_Latin-PROIEL la
ln -s UD_Latvian lv
ln -s UD_Norwegian no
ln -s UD_Old_Church_Slavonic cu
ln -s UD_Persian fa
ln -s UD_Portuguese pt
ln -s UD_Romanian ro
ln -s UD_Russian-SynTagRus ru
ln -s UD_Slovenian sl
ln -s UD_Spanish-AnCora es
ln -s UD_Swedish sv
ln -s UD_Tamil ta
ln -s UD_Turkish tr

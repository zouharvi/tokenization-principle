#!/usr/bin/bash

function balance_to_num {
    T=$(sed 's/k/000/g' <<<"$1")
    echo $T
}

for BALANCE_NAME in "300k-300k" "500k-100k" "100k-500k" "600k-000k" "000k-600k"; do
    BALANCE=$(balance_to_num $BALANCE_NAME)
    IFS='-' read -r -a BALANCE <<< "${BALANCE}";
    BALANCE1="${BALANCE[0]}"
    BALANCE2="${BALANCE[1]}"

    echo "${BALANCE1} --- ${BALANCE2}"

    DIR="data/tokenizer_bpe/${BALANCE_NAME}/"
    mkdir -p $DIR
    
    for LANGS in "de-en" "en-de"; do
        IFS='-' read -r -a LANGS <<< "${LANGS}";
        LANG1="${LANGS[0]}"
        LANG2="${LANGS[1]}"

        ORIG_DIR="data/tokenizer_bpe/${BALANCE_NAME}";
        TEXT_DIR="data_bin/tokenizer_bpe/${LANG1}-${LANG2}/${BALANCE_NAME}";
        mkdir -p ${TEXT_DIR};

        sbatch --time=0-1 --ntasks=30 --mem-per-cpu=1G \
            --job-name="preprocess_${BALANCE_NAME}.${LANG1}-${LANG2}" \
            --output="logs/preprocess_${BALANCE_NAME}.${LANG1}-${LANG2}.log" \
            --wrap="fairseq-preprocess --source-lang $LANG1 --target-lang $LANG2 \
                --trainpref $ORIG_DIR/train --validpref $ORIG_DIR/dev --testpref $ORIG_DIR/test  \
                --destdir $TEXT_DIR \
                --bpe fastbpe \
                --joined-dictionary \
                --tokenizer moses \
                --workers 30 \
            ";
    done;
done
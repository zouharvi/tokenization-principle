#!/usr/bin/bash

for VOCAB_SIZE in "4096" "8192" "16384"; do
    for BPE_MODEL in \
        "greedy" "antigreedy" \
        "random_0" "random_1" "random_2" "random_3" "random_4"; \
    do
        BPE_NAME="${BPE_MODEL}_${VOCAB_SIZE}"

        for LANGS in "en-de" "de-en"; do
            IFS='-' read -r -a LANGS <<< "${LANGS}";
            LANG1="${LANGS[0]}"
            LANG2="${LANGS[1]}"

            echo "Creating ${LANGS} data BPE'd by ${BPE_NAME}";
            TEXT_DIR="data_bin/CCrawl.${LANG1}-${LANG2}/${BPE_NAME}";
            mkdir -p ${TEXT_DIR};
        
            # take only 1M for train and 50k for eval
            head -n 1000000 "data_bpe/CCrawl.de-en/train.tok.${BPE_NAME}.${LANG1}" > "${TEXT_DIR}/train.${LANG1}";
            head -n 1000000 "data_bpe/CCrawl.de-en/train.tok.${BPE_NAME}.${LANG2}" > "${TEXT_DIR}/train.${LANG2}";
            head -n 50000 "data_bpe/CCrawl.de-en/dev.tok.${BPE_NAME}.${LANG1}" > "${TEXT_DIR}/dev.${LANG1}";
            head -n 50000 "data_bpe/CCrawl.de-en/dev.tok.${BPE_NAME}.${LANG2}" > "${TEXT_DIR}/dev.${LANG2}";
        done;

    done;
done;
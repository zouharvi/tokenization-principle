#!/usr/bin/bash

for VOCAB_SIZE in "4096" "8192" "16384"; do
    for BPE_MODEL in \
        "greedy" "antigreedy" \
        "random_0" "random_1" "random_2" "random_3" "random_4"; \
    do
        mkdir -p "data_bpe/CCrawl.de-en/"
        for SPLIT in "train" "dev"; do
            for LANG in "en" "de"; do
                NO_LINES="10000000";

                SIGNATURE="${SPLIT}_${LANG}_${BPE_MODEL}_${VOCAB_SIZE}"
                BPE_PATH="computed/${BPE_MODEL}_${VOCAB_SIZE}.bpe_model"
                SRC_DATA_PATH="data/CCrawl.de-en/${SPLIT}.tok.${LANG}"
                TGT_DATA_PATH="data_bpe/CCrawl.de-en/${SPLIT}.tok.${BPE_MODEL}_${VOCAB_SIZE}.${LANG}"

                if test -f "${BPE_PATH}" && ! test -f "${TGT_DATA_PATH}"; then
                    echo "Submitting ${SIGNATURE}";
                    sbatch --time=1-0 --ntasks=50 --mem-per-cpu=1G \
                        --output="logs/apply_bpe_${SIGNATURE}.log" \
                        --job-name="apply_bpe_${SIGNATURE}" \
                        --wrap="python3 ./src/apply_bpe.py \
                            --input ${SRC_DATA_PATH} \
                            --output ${TGT_DATA_PATH} \
                            --vocab-input ${BPE_PATH} \
                            --number-of-lines ${NO_LINES} \
                            --logfile computed/apply_bpe_all.jsonl \
                        ";
                else
                    if ! test -f "${BPE_PATH}"; then
                        echo "Skipping ${SIGNATURE} because ${BPE_PATH} does not exit"; 
                    elif test -f "${TGT_DATA_PATH}"; then
                        echo "Skipping ${SIGNATURE} because ${TGT_DATA_PATH} already exists"; 
                    else
                        echo "Skipping ${SIGNATURE} because ${BPE_PATH} does not exit and ${TGT_DATA_PATH} already exists"; 
                    fi;
                fi 
            done;
        done;
    done;
done;

#!/usr/bin/bash

GLOBAL_PARAMS="--number-of-lines 10000 -vs 2048 -i data/CCrawl.de-en/train.tok.en"
time ./src/fit_bpe.py $GLOBAL_PARAMS -vo computed/small/greedy_beamsearch.bpe_merges -m greedy_beamsearch

time ./src/fit_bpe.py $GLOBAL_PARAMS -vo computed/small/greedy.bpe_merges -m greedy

for LANG in "en" "de"; do
    # GLOBAL_PARAMS_APPLY="--number-of-lines 1000000 --logfile computed/apply_bpe_small.jsonl"
    GLOBAL_PARAMS_APPLY="--number-of-lines 10000 --logfile computed/apply_bpe_small.jsonl"
    METHOD="greedy_naive"
    for BPECODES in computed/small/*.bpe_merges; do
        ./src/apply_bpe.py $GLOBAL_PARAMS_APPLY \
            --method ${METHOD} \
            --vocab-input ${BPECODES} \
            --input "data/CCrawl.de-en/train.tok.${LANG}" \
            --output "/dev/null" &
    done;
done;



# for LANG in "en" "de"; do
#     GLOBAL_PARAMS_APPLY="--number-of-lines 1000000 --logfile computed/apply_bpe_small.jsonl"
#     METHOD="greedy_naive"
#     BPECODES="computed/small/greedy.bpe_merges"
#         ./src/apply_bpe.py $GLOBAL_PARAMS_APPLY \
#             --method ${METHOD} \
#             --vocab-input ${BPECODES} \
#             --input "data/CCrawl.de-en/train.tok.${LANG}" \
#             --output "/dev/null" &
# done;
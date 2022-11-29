#!/usr/bin/bash

GLOBAL_PARAMS="--number-of-lines 1000 -vs 512 -i data/demo.txt"

time ./src/fit_bpe.py $GLOBAL_PARAMS -vo computed/small/greedy_beamsearch.bpe_merges -m greedy_beamsearch
# real    2m19.900s
# user    2m19.845s
# sys     0m0.785s

time ./src/fit_bpe.py $GLOBAL_PARAMS -vo computed/small/greedy.bpe_merges -m greedy
# real    0m6.488s
# user    0m6.507s
# sys     0m0.666s

GLOBAL_PARAMS_APPLY="--number-of-lines 10000 --logfile computed/applybpe_small.jsonl"
METHOD="greedy_naive"
for BPECODES in computed/small/*.bpe_merges; do
    ./src/apply_bpe.py $GLOBAL_PARAMS_APPLY \
        --method ${METHOD} \
        --vocab-input ${BPECODES} \
        --input "data/demo.txt" \
        --output "/dev/null" &
done;
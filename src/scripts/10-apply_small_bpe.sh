#!/usr/bin/bash

GLOBAL_PARAMS="--number-of-lines 10000 --logfile computed/applybpe_small.jsonl"

for METHOD in "greedy_naive" "merge_operations"; do
    for BPECODES in computed/small/*.bpe_merges; do
        for LANG in "en" "de"; do
            for SPLIT in "dev" "train"; do
                ./src/apply_bpe.py $GLOBAL_PARAMS \
                    --method ${METHOD} \
                    --vocab-input ${BPECODES} \
                    --input "data/CCrawl.de-en/${SPLIT}.tok.${LANG}" \
                    --output "/dev/null" &
            done;
            wait
        done;
    done;
done;
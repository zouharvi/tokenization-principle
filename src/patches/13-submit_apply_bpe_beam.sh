#!/usr/bin/bash

GLOBAL_PARAMS_APPLY="--number-of-lines 10000 --logfile computed/applybpe_small.jsonl"
METHOD="greedy_naive"
for BPECODES in computed/small/*.bpe_merges; do
    SIGNATURE="${METHOD}";
    sbatch --time=1-0 --ntasks=20 --mem-per-cpu=2G \
        --output="logs/apply_bpe_${SIGNATURE}.log" \
        --job-name="apply_bpe_${SIGNATURE}" \
        --wrap="python3 ./src/apply_bpe.py $GLOBAL_PARAMS_APPLY \
        --method ${METHOD} \
        --vocab-input ${BPECODES} \
        --input data/demo.txt \
        --output /dev/null"
done;
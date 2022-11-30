#!/usr/bin/bash

# TODO: test also only 10k which is the data on which it was optimized
GLOBAL_PARAMS_APPLY="--number-of-lines 1000000 --logfile computed/apply_bpe_beam_mergeoperations.jsonl"
METHOD="merge_operations"
SPLIT="train"
for BPECODES in computed/small/*.bpe_merges; do
    for LANG in "en" "de"; do
        BPECODES_BASE=$(basename ${BPECODES})
        SIGNATURE="${METHOD}_${LANG}_${BPECODES_BASE}";
        echo "Launching ${SIGNATURE}"
        sbatch --time=1-0 --ntasks=20 --mem-per-cpu=2G \
            --output="logs/apply_bpe_${BPECODES_BASE}.log" \
            --job-name="apply_bpe_${BPECODES_BASE}" \
            --wrap="python3 ./src/apply_bpe.py $GLOBAL_PARAMS_APPLY \
            --method ${METHOD} \
            --vocab-input ${BPECODES} \
            --input \"data/CCrawl.de-en/${SPLIT}.tok.${LANG}\" \
            --output /dev/null"
    done;
done;
#!/usr/bin/bash

VOCAB_SIZE="1024"
LINE_COUNT="2000"
GLOBAL_PARAMS="--number-of-lines ${LINE_COUNT} -vs ${VOCAB_SIZE} -i data/CCrawl.de-en/train.tok.en"
time ./src/fit_bpe.py $GLOBAL_PARAMS -vo computed/small/greedy_v${VOCAB_SIZE}_l5k.bpe_merges -m greedy

time ./src/fit_bpe.py $GLOBAL_PARAMS -vo computed/small/greedy_beamsearch_${VOCAB_SIZE}_old.bpe_merges -m greedy_beamsearch_old --beam-n 3
time ./src/fit_bpe.py $GLOBAL_PARAMS -vo computed/small/greedy_beamsearch_${VOCAB_SIZE}.bpe_merges -m greedy_beamsearch --beam-n 3

time ./src/fit_bpe.py $GLOBAL_PARAMS -vo computed/small/greedy_v${VOCAB_SIZE}.bpe_merges -m greedy

for LANG in "de"; do
    GLOBAL_PARAMS_APPLY="--number-of-lines 1000000 --logfile computed/apply_bpe_small.jsonl"
    METHOD="greedy_naive"
    for BPECODES in computed/small/*.bpe_merges; do
        ./src/apply_bpe.py $GLOBAL_PARAMS_APPLY \
            --method ${METHOD} \
            --vocab-input ${BPECODES} \
            --input "data/CCrawl.de-en/train.tok.${LANG}" \
            --output "/dev/null" &
    done;
done;

# METHOD="merge_operations"
METHOD="greedy_naive"
GLOBAL_PARAMS_APPLY="--number-of-lines 100000 --logfile computed/apply_bpe_small.jsonl"
for LANG in "en" "de"; do
    # BPECODES="computed/small/greedy_v${VOCAB_SIZE}_l5k.bpe_merges"
    BPECODES="computed/small/greedy_beamsearch_v${VOCAB_SIZE}_l2k_n1000.bpe_merges"
    ./src/apply_bpe.py $GLOBAL_PARAMS_APPLY \
        --method ${METHOD} \
        --vocab-input ${BPECODES} \
        --input "data/CCrawl.de-en/train.tok.${LANG}" \
        --output "/dev/null";
done;

# speed
N="1000"
time_start=`date +%s`
./src/fit_bpe.py $GLOBAL_PARAMS \
    -vo /dev/null \
    -m greedy_beamsearch_new \
    -vo computed/small/greedy_beamsearch_v${VOCAB_SIZE}_l2k_n${N}.bpe_merges \
    --beam-n ${N}
time_end=`date +%s`
echo "n${N}" `expr $time_end - $time_start` "s"

for N in "1" "2" "3" "4" "5" "6" "7" "8" "9" "10"; do
done
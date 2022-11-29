#!/usr/bin/bash

GLOBAL_PARAMS="--number-of-lines 10000 -vs 2048"

./src/fit_bpe.py $GLOBAL_PARAMS -vo computed/small/antigreedy.bpe_merges -m antigreedy &
./src/fit_bpe.py $GLOBAL_PARAMS -vo computed/small/greedy.bpe_merges -m greedy &

# 2nd, 4th, 8th, ...
for N in 1 3 7 15 31 63; do
    ./src/fit_bpe.py $GLOBAL_PARAMS -vo computed/small/greedyalmost_${N}.bpe_merges -m greedyalmost --greedy-n ${N} &
done;

# sync up
wait

for i in 0 1 2 3 4; do
    ./src/fit_bpe.py $GLOBAL_PARAMS -vo computed/small/random_uni_${i}.bpe_merges -m random --randomness-dist uniform --seed ${i} &
done

wait

for TEMP_ALL in "1-1" "2-2" "4-4" "0.5-05" "0.25-025"; do
    IFS='-' read -r -a TEMP_ALL <<< "${TEMP_ALL}";
    TEMP_VAL="${TEMP_ALL[0]}"
    TEMP_TEXT="${TEMP_ALL[1]}"

    for i in 0 1 2 3 4; do
        ./src/fit_bpe.py $GLOBAL_PARAMS \
            -vo computed/small/random_softmax_t${TEMP_TEXT}_${i}.bpe_merges \
            -m random --randomness-dist softmax \
            --randomness-temp ${TEMP_VAL} --seed ${i} &
    done
    wait
done
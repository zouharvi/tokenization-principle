#!/usr/bin/bash

for BEAM_N_EXPAND in "5" "10"; do
for BEAM_N in "1" "2" "3" "5" "10"; do
    SIGNATURE="greedy_beamsearch_n${BEAM_N}_e${BEAM_N_EXPAND}"
    echo "Submitting ${SIGNATURE}";
    sbatch --time=1-0 --ntasks=20 --mem-per-cpu=2G \
        --output="logs/fit_bpe_${SIGNATURE}.log" \
        --job-name="fit_bpe_${SIGNATURE}" \
        --wrap="python3 ./src/fit_bpe.py \
            --model greedy_beamsearch \
            --vocab-output computed/small/greedy_beamsearch_n${BEAM_N}_e${BEAM_N_EXPAND}.bpe_merges \
            --vocab-size 2048 \
            --beam-n ${BEAM_N} \
            --beam-n-expand ${BEAM_N_EXPAND} \
            --number-of-lines 10000 \
        ";
done;
done;


SIGNATURE="greedy_small"
echo "Submitting ${SIGNATURE}";
sbatch --time=1-0 --ntasks=20 --mem-per-cpu=2G \
    --output="logs/fit_bpe_${SIGNATURE}.log" \
    --job-name="fit_bpe_${SIGNATURE}" \
    --wrap="python3 ./src/fit_bpe.py \
        --model greedy \
        --vocab-output computed/small/greedy.bpe_merges \
        --vocab-size 2048 \
        --number-of-lines 10000 \
    ";
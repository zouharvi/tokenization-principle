#!/usr/bin/bash

for VOCAB_SIZE in "4096" "8192" "16384"; do
    for BEAM_N_EXPAND in "3" "5" "10"; do
    for BEAM_N in "2" "3" "5" "10" "20"; do
        SIGNATURE="greedy_beamsearch_n${BEAM_N}_e${BEAM_N_EXPAND}_v${VOCAB_SIZE}"
        echo "Submitting ${SIGNATURE}";
        sbatch --time=4-0 --ntasks=5 --mem-per-cpu=2G \
            --output="logs/fit_bpe_${SIGNATURE}.log" \
            --job-name="fit_bpe_${SIGNATURE}" \
            --wrap="time python3 ./src/fit_bpe.py \
                --model greedy_beamsearch \
                --vocab-output computed/small/greedy_beamsearch_n${BEAM_N}_e${BEAM_N_EXPAND}_v${VOCAB_SIZE}.bpe_merges \
                --vocab-size ${VOCAB_SIZE} \
                --beam-n ${BEAM_N} \
                --beam-n-expand ${BEAM_N_EXPAND} \
                --number-of-lines 20000 \
            ";
    done;
    done;


    SIGNATURE="greedy_small"
    echo "Submitting ${SIGNATURE}";
    sbatch --time=1-0 --ntasks=5 --mem-per-cpu=2G \
        --output="logs/fit_bpe_${SIGNATURE}.log" \
        --job-name="fit_bpe_${SIGNATURE}" \
        --wrap="time python3 ./src/fit_bpe.py \
            --model greedy \
            --vocab-output computed/small/greedy_v${VOCAB_SIZE}.bpe_merges \
            --vocab-size ${VOCAB_SIZE} \
            --number-of-lines 20000 \
        ";
done;
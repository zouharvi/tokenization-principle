#!/usr/bin/bash

for VOCAB_SIZE in "4096" "8192" "16384"; do
    for BPE_MODEL in "greedy" "random" "antigreedy"; do
        echo "Submitting ${BPE_MODEL}_${VOCAB_SIZE}";
        sbatch --time=1-0 --ntasks=20 --mem-per-cpu=2G \
            --output="logs/fit_bpe_${BPE_MODEL}_${VOCAB_SIZE}.log" \
            --job-name="fit_bpe_${BPE_MODEL}_${VOCAB_SIZE}" \
            --wrap="python3 ./src/fit_bpe.py \
                --model $BPE_MODEL \
                --vocab-output computed/$BPE_MODEL.bpe_model \
                --vocab-size $VOCAB_SIZE \
                --number-of-lines 100000 \
            ";
    done;
done;
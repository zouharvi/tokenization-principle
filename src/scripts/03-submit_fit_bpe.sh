#!/usr/bin/bash

# Create all the tokenizations

for VOCAB_SIZE in "4096" "8192" "16384"; do
    for BPE_MODEL in "greedy" "antigreedy"; do
        SIGNATURE="${BPE_MODEL}_${VOCAB_SIZE}"
        echo "Submitting ${SIGNATURE}";
        sbatch --time=1-0 --ntasks=10 --mem-per-cpu=4G \
            -J "fit_bpe_${SIGNATURE}" \
            --output=logs/%x.out --error=logs/%x.err \
            --wrap="python3 ./src/fit_bpe.py \
                --model $BPE_MODEL \
                --vocab-output computed/${BPE_MODEL}_${VOCAB_SIZE}.bpe_model \
                --vocab-size $VOCAB_SIZE \
                --number-of-lines 100000 \
            ";
    done;

    BPE_MODEL="random"
    for SEED in "0" "1" "2" "3" "4"; do
        SIGNATURE="${BPE_MODEL}_${SEED}_${VOCAB_SIZE}"
        echo "Submitting ${SIGNATURE}";
        sbatch --time=1-0 --ntasks=10 --mem-per-cpu=4G \
            -J "fit_bpe_${SIGNATURE}" \
            --output=logs/%x.out --error=logs/%x.err \
            --wrap="python3 ./src/fit_bpe.py \
                --model $BPE_MODEL \
                --seed $SEED \
                --vocab-output computed/${BPE_MODEL}_${SEED}_${VOCAB_SIZE}.bpe_model \
                --vocab-size $VOCAB_SIZE \
                --number-of-lines 100000 \
            ";
    done;
done;

VOCAB_SIZE="4096"
BPE_MODEL="greedy"
SIGNATURE="${BPE_MODEL}_${VOCAB_SIZE}"
echo "Submitting ${SIGNATURE}";
sbatch --time=0-4 --ntasks=10 --mem-per-cpu=4G \
    -J "fit_bpe_${SIGNATURE}" \
    --output=logs/%x.out --error=logs/%x.err \
    --wrap="python3 ./src/fit_bpe.py \
        --model $BPE_MODEL \
        --vocab-output computed/${BPE_MODEL}_${VOCAB_SIZE}.bpe_model \
        --vocab-size $VOCAB_SIZE \
        --number-of-lines 100000 \
    ";
#!/usr/bin/bash

for VOCAB_SIZE in "2000" "4000"; do
    for BPE_MODEL in "greedy_pmi" "greedy" "random_pmi_freq"; do
        SIGNATURE="${BPE_MODEL}_${VOCAB_SIZE}"
        echo "Submitting ${SIGNATURE}";
        mkdir -p "data/${BPE_MODEL}/${VOCAB_SIZE}"

        python3 ./src/fit_bpe.py \
                --model $BPE_MODEL \
                --vocab-output data/${BPE_MODEL}/${VOCAB_SIZE}/model.bpe \
                --vocab-size $VOCAB_SIZE \
                --number-of-lines 100000 \
            ;

        # sbatch --time=1-0 --ntasks=20 --mem-per-cpu=2G \
        # --output="logs/fit_bpe_${SIGNATURE}.log" \
        # --job-name="fit_bpe_${SIGNATURE}" \
        # --wrap="python3 ./src/fit_bpe.py \
        #         --model $BPE_MODEL \
        #         --vocab-output data/${BPE_MODEL}/${VOCAB_SIZE}/model.bpe \
        #         --vocab-size $VOCAB_SIZE \
        #         --number-of-lines 100000 \
        #     ";
    done;
done;
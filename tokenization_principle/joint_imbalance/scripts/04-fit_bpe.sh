#!/usr/bin/bash

for VOCAB_SIZE in "2000" "4000"; do
    for BPE_MODEL in "greedy_pmi" "greedy" "random_pmi_freq"; do
        SIGNATURE="${BPE_MODEL}_${VOCAB_SIZE}"
        echo "Submitting ${SIGNATURE}";
        mkdir -p "data/${BPE_MODEL}/${VOCAB_SIZE}"

        python3 -m tokenization_principle.joint_imbalance.fit_bpe \
                --model $BPE_MODEL \
                --vocab-output data/${BPE_MODEL}/${VOCAB_SIZE}/model.bpe \
                --vocab-size $VOCAB_SIZE \
                --number-of-lines 100000 \
            ;

        # sbatch --time=1-0 --ntasks=20 --mem-per-cpu=2G \
        # -J "fit_bpe_${SIGNATURE}" \
        # --output=logs/%x.out --error=logs/%x.err \
        # --wrap="python3 -m tokenization_principle.joint_imbalance.fit_bpe \
        #         --model $BPE_MODEL \
        #         --vocab-output data/${BPE_MODEL}/${VOCAB_SIZE}/model.bpe \
        #         --vocab-size $VOCAB_SIZE \
        #         --number-of-lines 100000 \
        #     ";
    done;
done;
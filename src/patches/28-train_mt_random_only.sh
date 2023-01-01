#!/usr/bin/bash

temperature_name() {
    T=$(sed 's/^-/m/g' <<<"$1")
    T=$(sed 's/^0./0/g' <<<"$T")
    T=$(sed 's/m0./m0/g' <<<"$T")
    echo $T
}

vocab_size_name() {
    T=$(sed 's/000$/k/g' <<<"$1")
    echo $T
}

SEED="2"
for VOCAB_SIZE in "2000" "4000" "8000" "16000" "32000"; do
for TEMPERATURE in "0.05" "0.2" "0.4" "0.9" "100" "-100" "-0.9" "-0.4" "-0.2" "-0.00001"; do
    MODEL="bpe_random"
    TEMPERATURE_NAME=$(temperature_name $TEMPERATURE)
    VOCAB_SIZE_NAME=$(vocab_size_name $VOCAB_SIZE)
    SIGNATURE="s${SEED}_t${TEMPERATURE_NAME}_v${VOCAB_SIZE_NAME}"

    MODEL="bpe_random"
    # for LANGS in "en-de" "de-en"; do
    for LANGS in "de-en"; do
        IFS='-' read -r -a LANGS <<< "${LANGS}";
        LANG1="${LANGS[0]}"
        LANG2="${LANGS[1]}"

        echo "Submitting ${SIGNATURE}";
        TEXT_DIR="data_bin/${LANG1}-${LANG2}/model_${MODEL}/${TEMPERATURE_NAME}_${VOCAB_SIZE_NAME}";
        CHECKPOINT_DIR="data_bin_ckpt/${LANG1}-${LANG2}/model_${MODEL}/${SIGNATURE}";
        mkdir -p ${CHECKPOINT_DIR}

        sbatch --time=07-00 --ntasks=8 --mem-per-cpu=4G --gpus=1 \
            --job-name="train_mt_${SIGNATURE}" \
            --output="logs/train_mt_${SIGNATURE}.log" \
            --wrap="CUDA_VISIBLE_DEVICES=0 fairseq-train \
                $TEXT_DIR \
                --source-lang $LANG1 --target-lang $LANG2 \
                --no-progress-bar \
                --log-interval 2000 \
                --arch transformer_iwslt_de_en --share-decoder-input-output-embed \
                --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.5 \
                --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
                --dropout 0.3 --weight-decay 0.0001 \
                --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
                --max-tokens 3072 \
                --eval-bleu \
                --patience 5 \
                --save-dir \"$CHECKPOINT_DIR/checkpoints\" \
                --eval-bleu-args '{\"beam\": 5, \"max_len_a\": 1.2, \"max_len_b\": 10}' \
                --eval-bleu-detok moses \
                --eval-bleu-remove-bpe \
                --bpe fastbpe \
                --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
                --seed ${SEED} \
                --amp \
            "
                # --eval-bleu-print-samples \
    done;
done;
done;
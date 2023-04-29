#!/usr/bin/bash

for SEED in "0" "1" "2" "3" "4" "5" "6"; do
for BALANCE_NAME in "300k-300k" "500k-100k" "100k-500k" "600k-000k" "000k-600k"; do
    for LANGS_NAME in "de-en" "en-de"; do
        IFS='-' read -r -a LANGS <<< "${LANGS_NAME}";
        LANG1="${LANGS[0]}"
        LANG2="${LANGS[1]}"
        SIGNATURE="s${SEED}_${LANGS_NAME}_${BALANCE_NAME}"

        echo "Submitting ${SIGNATURE}";
        TEXT_DIR="data_bin/tokenizer_bpe/${LANG1}-${LANG2}/${BALANCE_NAME}";
        CHECKPOINT_DIR="data_bin_ckpt/tokenizer_bpe/${SIGNATURE}/";
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
                --save-dir \"$CHECKPOINT_DIR\" \
                --eval-bleu-args '{\"beam\": 5, \"max_len_a\": 1.2, \"max_len_b\": 10}' \
                --eval-bleu-detok moses \
                --eval-bleu-remove-bpe \
                --bpe fastbpe \
                --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
                --seed ${SEED} \
                --no-epoch-checkpoints \
                --amp \
            "
    done;
done;
done;
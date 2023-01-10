#!/usr/bin/bash

SEED="1"
for VOCAB_SIZE_NAME in "4k" "8k" "16k"; do
for TRAIN_LINES_NAME in "2k" "8k" "100k"; do
for MODEL in \
    "morfessor" "lzw"\
    "tokenizer_unigram" "tokenizer_wordpiece" "tokenizer_bpe"; \
do
    SIGNATURE="s${SEED}_l${TRAIN_LINES_NAME}_v${VOCAB_SIZE_NAME}"

    # for LANGS in "en-de" "de-en"; do
    for LANGS in "de-en"; do
        IFS='-' read -r -a LANGS <<< "${LANGS}";
        LANG1="${LANGS[0]}"
        LANG2="${LANGS[1]}"

        echo "Submitting ${MODEL} ${SIGNATURE}";
        TEXT_DIR="data_bin/${LANG1}-${LANG2}/model_${MODEL}/l${TRAIN_LINES_NAME}_${VOCAB_SIZE_NAME}";
        CHECKPOINT_DIR="data_bin_ckpt/${LANG1}-${LANG2}/model_${MODEL}/${SIGNATURE}";
        mkdir -p ${CHECKPOINT_DIR}

        sbatch --time=07-00 --ntasks=8 --mem-per-cpu=4G --gpus=1 \
            --job-name="train_mt_${MODEL}_${SIGNATURE}" \
            --output="logs/train_mt_${MODEL}_${SIGNATURE}.log" \
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
                --keep-last-epochs 0 \
                --amp \
            "
    done;
done;
done;
done;
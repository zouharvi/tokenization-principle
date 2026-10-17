#!/usr/bin/bash

SEED="5"
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
        DATA_DIR="data/model_${MODEL}/l${TRAIN_LINES_NAME}_${VOCAB_SIZE_NAME}";
        CHECKPOINT_DIR="data_bin_ckpt/${LANG1}-${LANG2}/model_${MODEL}/${SIGNATURE}";
        mkdir -p "${CHECKPOINT_DIR}/checkpoints"

        sbatch --time=07-00 --ntasks=8 --mem-per-cpu=4G --gpus=1 \
            -J "train_mt_${MODEL}_${SIGNATURE}" \
            --output=logs/%x.out --error=logs/%x.err \
            --wrap="CUDA_VISIBLE_DEVICES=0 python3 ./src/train_mt_pytorch.py \
                --train-src \"${DATA_DIR}/train.${LANG1}\" \
                --train-tgt \"${DATA_DIR}/train.${LANG2}\" \
                --valid-src \"${DATA_DIR}/dev.${LANG1}\" \
                --valid-tgt \"${DATA_DIR}/dev.${LANG2}\" \
                --test-src \"${DATA_DIR}/test.${LANG1}\" \
                --test-tgt \"${DATA_DIR}/test.${LANG2}\" \
                --save-dir \"${CHECKPOINT_DIR}/checkpoints\" \
                --max-tokens 12288 \
                --log-interval 2000 \
                --embed-dim 512 \
                --ffn-dim 1024 \
                --encoder-layers 6 \
                --decoder-layers 6 \
                --num-heads 4 \
                --lr 5e-4 \
                --warmup-updates 4000 \
                --dropout 0.3 \
                --weight-decay 0.0001 \
                --clip-norm 0.5 \
                --label-smoothing 0.1 \
                --patience 5 \
                --seed ${SEED} \
            "
    done;
done;
done;
done;
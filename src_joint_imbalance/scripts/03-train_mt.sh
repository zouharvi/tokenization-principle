#!/usr/bin/bash

for SEED in "0" "1" "2" "3" "4" "5" "6"; do
for BALANCE_NAME in "300k-300k" "500k-100k" "100k-500k" "600k-000k" "000k-600k"; do
    for LANGS_NAME in "de-en" "en-de"; do
        IFS='-' read -r -a LANGS <<< "${LANGS_NAME}";
        LANG1="${LANGS[0]}"
        LANG2="${LANGS[1]}"
        SIGNATURE="s${SEED}_${LANGS_NAME}_${BALANCE_NAME}"

        echo "Submitting ${SIGNATURE}";
        DATA_DIR="data/tokenizer_bpe/${BALANCE_NAME}";
        CHECKPOINT_DIR="data_bin_ckpt/tokenizer_bpe/${SIGNATURE}/";
        mkdir -p "${CHECKPOINT_DIR}/checkpoints"
        mkdir -p "logs"

        sbatch --time=07-00 --ntasks=8 --mem-per-cpu=4G --gpus=1 \
            --job-name="train_mt_${SIGNATURE}" \
            --output="logs/train_mt_${SIGNATURE}.log" \
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
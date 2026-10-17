#!/usr/bin/bash

# Unified MT Training Submission Script
# Trains Transformer models for all prepared tokenizations in data/prepared/

SEED="5"
LANGS_LIST=("de-en" "cs-en")

mkdir -p logs

for LANGS in "${LANGS_LIST[@]}"; do
    IFS='-' read -r -a L <<< "${LANGS}"
    LANG1="${L[0]}"
    LANG2="${L[1]}"

    for MODEL_DIR in data/prepared/${LANGS}/*; do
        [ -d "${MODEL_DIR}" ] || continue
        SIGNATURE=$(basename "${MODEL_DIR}")
        CHECKPOINT_DIR="${MODEL_DIR}/checkpoints"
        mkdir -p "${CHECKPOINT_DIR}"

        echo "Submitting MT training: ${LANGS} ${SIGNATURE}"
        sbatch --time=07-00 --ntasks=8 --mem-per-cpu=4G --gpus=1 \
            --job-name="train_mt_${LANGS}_${SIGNATURE}" \
            --output="logs/train_mt_${LANGS}_${SIGNATURE}.log" \
            --wrap="CUDA_VISIBLE_DEVICES=0 python3 -m tokenization_principle.train_mt_pytorch \
                --train-src \"${MODEL_DIR}/train.${LANG1}\" \
                --train-tgt \"${MODEL_DIR}/train.${LANG2}\" \
                --valid-src \"${MODEL_DIR}/dev.${LANG1}\" \
                --valid-tgt \"${MODEL_DIR}/dev.${LANG2}\" \
                --save-dir \"${CHECKPOINT_DIR}\" \
                --max-tokens 12288 \
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
    done
done

#!/usr/bin/bash

# Unified MT Training Submission Script
# Trains Transformer models for all prepared tokenizations in data/prepared/

SEEDS=("0")
LANGS_LIST=("de-en" "cs-en" "zh-en")

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

        for SEED in "${SEEDS[@]}"; do
            echo "Submitting MT training: ${LANGS} ${SIGNATURE} (seed ${SEED})"
            sbatch --time=07-00 --ntasks=8 --mem-per-cpu=8G --gpus=rtx_4090:1 \
                --job-name="train_mt_${LANGS}_${SIGNATURE}_s${SEED}" \
                --output="logs/train_mt_${LANGS}_${SIGNATURE}_s${SEED}.log" \
                --wrap="python3 -u -m tokenization_principle.train_mt_pytorch \
                    --device cuda \
                    --max-train-lines 10000000 \
                    --train-src \"${MODEL_DIR}/train.${LANG1}\" \
                    --train-tgt \"${MODEL_DIR}/train.${LANG2}\" \
                    --valid-src \"${MODEL_DIR}/dev.${LANG1}\" \
                    --valid-tgt \"${MODEL_DIR}/dev.${LANG2}\" \
                    --save-dir \"${CHECKPOINT_DIR}\" \
                    --max-tokens 12288 \
                    --embed-dim 512 \
                    --ffn-dim 2048 \
                    --encoder-layers 6 \
                    --decoder-layers 6 \
                    --num-heads 8 \
                    --lr 1e-3 \
                    --warmup-updates 1000 \
                    --dropout 0.3 \
                    --weight-decay 0.0001 \
                    --clip-norm 0.5 \
                    --label-smoothing 0.1 \
                    --patience 1 \
                    --seed ${SEED} \
                "
        done
    done
done


sbatch --time=01-00 --ntasks=8 --mem-per-cpu=8G --gpus=rtx_4090:1 \
    --job-name="train_mt_de-en_unigram_v32000_s0" \
    --output="logs/train_mt_de-en_unigram_v32000_s0.log" \
    --wrap="python3 -u -m tokenization_principle.train_mt_pytorch \
        --device cuda \
        --max-train-lines 10000000 \
        --train-src \"data/prepared/de-en/unigram_v32000/train.de\" \
        --train-tgt \"data/prepared/de-en/unigram_v32000/train.en\" \
        --valid-src \"data/prepared/de-en/unigram_v32000/dev.de\" \
        --valid-tgt \"data/prepared/de-en/unigram_v32000/dev.en\" \
        --save-dir \"data/prepared/de-en/unigram_v32000/checkpoints\" \
        --max-tokens 12288 \
        --embed-dim 512 \
        --ffn-dim 2048 \
        --encoder-layers 6 \
        --decoder-layers 6 \
        --num-heads 8 \
        --lr 1e-3 \
        --warmup-updates 1000 \
        --dropout 0.3 \
        --weight-decay 0.0001 \
        --clip-norm 0.5 \
        --label-smoothing 0.1 \
        --patience 1 \
        --seed 0 \
    "


sbatch --time=01-00 --ntasks=8 --mem-per-cpu=8G --gpus=rtx_4090:1 \
    --job-name="train_mt_de-en_unigram_v4000_s0" \
    --output="logs/train_mt_de-en_unigram_v4000_s0.log" \
    --wrap="python3 -u -m tokenization_principle.train_mt_pytorch \
        --device cuda \
        --max-train-lines 10000000 \
        --train-src \"data/prepared/de-en/unigram_v4000/train.de\" \
        --train-tgt \"data/prepared/de-en/unigram_v4000/train.en\" \
        --valid-src \"data/prepared/de-en/unigram_v4000/dev.de\" \
        --valid-tgt \"data/prepared/de-en/unigram_v4000/dev.en\" \
        --save-dir \"data/prepared/de-en/unigram_v4000/checkpoints\" \
        --max-tokens 12288 \
        --embed-dim 512 \
        --ffn-dim 2048 \
        --encoder-layers 6 \
        --decoder-layers 6 \
        --num-heads 8 \
        --lr 1e-3 \
        --warmup-updates 1000 \
        --dropout 0.3 \
        --weight-decay 0.0001 \
        --clip-norm 0.5 \
        --label-smoothing 0.1 \
        --patience 1 \
        --seed 0 \
    "
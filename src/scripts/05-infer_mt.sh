#!/usr/bin/bash

# Unified MT Inference Submission Script
# Generates translations on the dev set for all trained models in data/prepared/

LANGS_LIST=("de-en" "cs-en")

mkdir -p logs

for LANGS in "${LANGS_LIST[@]}"; do
    IFS='-' read -r -a L <<< "${LANGS}"
    LANG1="${L[0]}"
    LANG2="${L[1]}"

    for MODEL_DIR in data/prepared/${LANGS}/*; do
        [ -d "${MODEL_DIR}" ] || continue
        SIGNATURE=$(basename "${MODEL_DIR}")
        MODEL_PATH="${MODEL_DIR}/checkpoints/checkpoint_best.pt"
        DATAOUT_PATH="${MODEL_DIR}/dev_out.${LANG2}"
        INPUT_PATH="${MODEL_DIR}/dev.${LANG1}"

        [ -f "${MODEL_PATH}" ] || continue

        echo "Submitting MT inference: ${LANGS} ${SIGNATURE}"
        sbatch --time=00-01 --ntasks=4 --mem-per-cpu=4G --gpus=1 \
            --job-name="infer_mt_${LANGS}_${SIGNATURE}" \
            --output="logs/infer_mt_${LANGS}_${SIGNATURE}.log" \
            --wrap="CUDA_VISIBLE_DEVICES=0 python3 ./src/infer_mt_pytorch.py \
                --model-path \"${MODEL_PATH}\" \
                --input \"${INPUT_PATH}\" \
                --output \"${DATAOUT_PATH}\" \
                --batch-size 256 \
            "
    done
done

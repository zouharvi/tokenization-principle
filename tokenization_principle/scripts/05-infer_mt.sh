#!/usr/bin/bash

# Unified MT Inference Submission Script
# Generates translations on the dev set for all trained models in data/prepared/

LANGS_LIST=("de-en" "cs-en" "zh-en")

mkdir -p logs

for LANGS in "${LANGS_LIST[@]}"; do
    IFS='-' read -r -a L <<< "${LANGS}"
    LANG1="${L[0]}"
    LANG2="${L[1]}"

    for MODEL_DIR in data/prepared/${LANGS}/*; do
        [ -d "${MODEL_DIR}" ] || continue
        SIGNATURE=$(basename "${MODEL_DIR}")

        for MODEL_PATH in "${MODEL_DIR}/checkpoints"/checkpoint_best*.pt; do
            [ -f "${MODEL_PATH}" ] || continue
            CKPT_FILE=$(basename "${MODEL_PATH}" .pt)
            SEED_SUFFIX=$(echo "${CKPT_FILE}" | sed -E 's/^checkpoint_best(_s[0-9]+)?$/\1/')
            DATAOUT_PATH="${MODEL_DIR}/dev_out${SEED_SUFFIX}.${LANG2}"
            INPUT_PATH="${MODEL_DIR}/dev.${LANG1}"

            echo "Submitting MT inference: ${LANGS} ${SIGNATURE} (${CKPT_FILE})"
            sbatch --time=00-01 --ntasks=4 --mem-per-cpu=4G --gpus=1 \
                --job-name="infer_mt_${LANGS}_${SIGNATURE}${SEED_SUFFIX}" \
                --output="logs/infer_mt_${LANGS}_${SIGNATURE}${SEED_SUFFIX}.log" \
                --wrap="python3 -u -m tokenization_principle.infer_mt_pytorch \
                    --model-path \"${MODEL_PATH}\" \
                    --input \"${INPUT_PATH}\" \
                    --output \"${DATAOUT_PATH}\" \
                    --batch-size 256 \
                "
        done
    done
done

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
MODEL="bpe_random"

for LANGS in "de-en" "cs-en"; do
    IFS='-' read -r -a LANGS <<< "${LANGS}";
    LANG1="${LANGS[0]}"
    LANG2="${LANGS[1]}"

for VOCAB_SIZE in "4000" "8000" "16000" "32000"; do
for TEMPERATURE in "0.05" "0.4" "100" "-0.9" "-0.2"; do
    TEMPERATURE_NAME=$(temperature_name $TEMPERATURE)
    VOCAB_SIZE_NAME=$(vocab_size_name $VOCAB_SIZE)
    SIGNATURE="s${SEED}_t${TEMPERATURE_NAME}_v${VOCAB_SIZE_NAME}"

    echo "Submitting ${SIGNATURE}";
    DATA_DIR="data/model_${MODEL}/${LANG1}-${LANG2}/${TEMPERATURE_NAME}_${VOCAB_SIZE_NAME}";
    CHECKPOINT_DIR="data_bin_ckpt_repl/${LANG1}-${LANG2}/model_${MODEL}/${SIGNATURE}";
    mkdir -p "${CHECKPOINT_DIR}/checkpoints"
    mkdir -p "logs_repl"

    sbatch --time=07-00 --ntasks=3 --mem-per-cpu=10G --gpus=1 \
        --job-name="train_mt_${SIGNATURE} (${LANG1}-${LANG2})" \
        --output="logs_repl/train_mt_${SIGNATURE}_${LANG1}-${LANG2}.log" \
        --wrap="CUDA_VISIBLE_DEVICES=0 python3 ./src/train_mt_pytorch.py \
            --train-src \"${DATA_DIR}/train.${LANG1}\" \
            --train-tgt \"${DATA_DIR}/train.${LANG2}\" \
            --valid-src \"${DATA_DIR}/dev.${LANG1}\" \
            --valid-tgt \"${DATA_DIR}/dev.${LANG2}\" \
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
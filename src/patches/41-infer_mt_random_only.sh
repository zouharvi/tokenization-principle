#!/usr/bin/bash

function launch_mt_infer() {
    # $1 cs-en
    # $2 s1_t005_v16k
    # $3 data/model_bpe_random/cs-en/005_16k/dev.cs
    # $4 005_16k

    echo "Launching ${1} ${2} ${3} ${4}" 
    MODEL_PATH="data_bin_ckpt_repl/${1}/model_bpe_random/${2}/checkpoints/checkpoint_best.pt"
    TEXT_PATH="data_bin_repl/${1}/model_bpe_random/${4}/"
    DATAOUT_PATH="data_bin_ckpt_repl/${1}/model_bpe_random/${2}/dev_out.en"
    
    # we use fairseq-interactive because fairseq-generate had some issues
    # it shouldn't be that much slower because we're using buffe-size 100 (batching)
    # so the only part where we're slower is some piping and the binarization,
    # which is fairly lightweight
    awk 'length >= 1 && length <= 100' "${3}" > "${3}.small"
    sbatch --time=00-01 --ntasks=8 --mem-per-cpu=6G --gpus=1 \
    --job-name="infer_mt_${2}" \
    --output="logs_repl/infer_mt_${2}.log" \
    --wrap="CUDA_VISIBLE_DEVICES=0 
    fairseq-interactive \
        ${TEXT_PATH} \
        --path ${MODEL_PATH} \
        --beam 5 \
        --remove-bpe \
        --max-tokens 4096 \
        --tokenizer space \
        --seed 0 \
        --buffer-size 100 \
        --input ${3}.small \
        > ${DATAOUT_PATH} \
    "
}
# --gres=gpumem:12g \

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

# 
for SEED in "0" "1" "2"; do
for LANGS in "cs-en" "de-en"; do
    IFS='-' read -r -a LANGS <<< "${LANGS}";
    LANG1="${LANGS[0]}"
    LANG2="${LANGS[1]}"
# smaller vocab range
for VOCAB_SIZE in "4000" "8000" "16000" "32000"; do
# smaller temperature range
for TEMPERATURE in "0.05" "0.4" "100" "-0.9" "-0.2"; do
    TEMPERATURE_NAME=$(temperature_name $TEMPERATURE)
    VOCAB_SIZE_NAME=$(vocab_size_name $VOCAB_SIZE)
    SIGNATURE_FULL="s${SEED}_t${TEMPERATURE_NAME}_v${VOCAB_SIZE_NAME}"
    SIGNATURE_SMALL="${TEMPERATURE_NAME}_${VOCAB_SIZE_NAME}"
    
    PATH_IN="data/model_bpe_random/${LANG1}-${LANG2}/${SIGNATURE_SMALL}/dev.${LANG1}"
    launch_mt_infer "${LANG1}-${LANG2}" "${SIGNATURE_FULL}" "${PATH_IN}" "${SIGNATURE_SMALL}"
done;
done;
done;
done;

# tail -n 1 logs_repl/infer_mt_s*
# awk 'length >= 1 && length <= 100' "data/model_bpe_random/cs-en/005_4k/dev.cs" | wc -l

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

file_to_lang() {
    LANGS=$(sed 's/data\///g' <<<"$1")
    LANGS=$(sed 's/\/train.tok.en//g' <<<"$LANGS")
    LANGS=$(sed 's/.Crawl\.//g' <<<"$LANGS")
    echo $LANGS
}

SEED="1"
MODEL="bpe_random"
for FILES in \
    "data/CCrawl.de-en/train.tok.en^data/CCrawl.de-en/train.tok.de" \
    "data/PCrawl.zh-en/train.tok.en^data/PCrawl.zh-en/train.tok.zh"; do

    IFS='^' read -r -a FILES <<< "${FILES}";
    FILE1="${FILES[0]}"
    FILE2="${FILES[1]}"
# for VOCAB_SIZE in "2000" "4000" "8000" "16000" "32000" ; do
# for TEMPERATURE in "0.05" "0.2" "0.4" "0.9" "100" "-100" "-0.9" "-0.4" "-0.2" "-0.00001"; do
# smaller vocab range
for VOCAB_SIZE in "4000" "8000" "16000" "32000"; do
# smaller temperature range
for TEMPERATURE in "0.05" "0.4" "100" "-0.9" "-0.2"; do
    MODEL="bpe_random"
    TEMPERATURE_NAME=$(temperature_name $TEMPERATURE)
    VOCAB_SIZE_NAME=$(vocab_size_name $VOCAB_SIZE)
    SIGNATURE="s${SEED}_t${TEMPERATURE_NAME}_v${VOCAB_SIZE_NAME}"

    LANGS=$(file_to_lang $FILE1)
    IFS='-' read -r -a LANGS <<< "${LANGS}";
    LANG1="${LANGS[0]}"
    LANG2="${LANGS[1]}"

    echo "Submitting ${SIGNATURE}";
    TEXT_DIR="data_bin_repl/${LANG1}-${LANG2}/model_${MODEL}/${TEMPERATURE_NAME}_${VOCAB_SIZE_NAME}";
    CHECKPOINT_DIR="data_bin_ckpt_repl/${LANG1}-${LANG2}/model_${MODEL}/${SIGNATURE}";
    mkdir -p ${CHECKPOINT_DIR}

    sbatch --time=07-00 --ntasks=3 --mem-per-cpu=10G --gpus=1 \
        --job-name="train_mt_${SIGNATURE} (${LANG1}-${LANG2})" \
        --output="logs_repl/train_mt_${SIGNATURE}_${LANG1}-${LANG2}.log" \
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
            --keep-last-epochs 1 \
        "
    done;
done;
done;
#!/usr/bin/bash

for VOCAB_SIZE in "16384" "8192" "4096"; do
for BPE_MODEL in \
    "greedy" "antigreedy" \
    "random_0" "random_1" "random_2" "random_3" "random_4"; \
do
    BPE_NAME="${BPE_MODEL}_${VOCAB_SIZE}"

    for LANGS in "en-de" "de-en"; do
        IFS='-' read -r -a LANGS <<< "${LANGS}";
        LANG1="${LANGS[0]}"
        LANG2="${LANGS[1]}"

        # TODO: training joiend
        SIGNATURE="${BPE_NAME}.${LANG1}-${LANG2}";
        echo "Submitting ${SIGNATURE}";
        TEXT_DIR="data_bin/CCrawl.${LANG1}-${LANG2}/${BPE_NAME}";

        sbatch --time=07-00 --ntasks=8 --mem-per-cpu=4G --gpus=1 \
            --job-name="train_mt_${SIGNATURE}" \
            --output="logs/train_mt_${SIGNATURE}.log" \
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
                --max-tokens 4096 \
                --eval-bleu \
                --patience 10 \
                --save-dir \"$TEXT_DIR/checkpoints\" \
                --eval-bleu-args '{\"beam\": 5, \"max_len_a\": 1.2, \"max_len_b\": 10}' \
                --eval-bleu-detok moses \
                --eval-bleu-remove-bpe \
                --bpe fastbpe \
                --eval-bleu-print-samples \
                --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
            "
    done;
done;
done;

# fp16 is apparently not supported and will result in slower training
# --fp16 \
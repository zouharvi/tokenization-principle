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

        SIGNATURE="${BPE_NAME}.${LANG1}-${LANG2}";
        echo "Submitting ${SIGNATURE}";
        TEXT_DIR="data_bin/CCrawl.${LANG1}-${LANG2}/${BPE_NAME}";
        mkdir -p "${TEXT_DIR}/checkpoints"

        sbatch --time=07-00 --ntasks=8 --mem-per-cpu=4G --gpus=1 \
            --job-name="train_mt_${SIGNATURE}" \
            --output="logs/train_mt_${SIGNATURE}.log" \
            --wrap="CUDA_VISIBLE_DEVICES=0 python3 ./src/train_mt_pytorch.py \
                --train-src \"$TEXT_DIR/train.$LANG1\" \
                --train-tgt \"$TEXT_DIR/train.$LANG2\" \
                --valid-src \"$TEXT_DIR/dev.$LANG1\" \
                --valid-tgt \"$TEXT_DIR/dev.$LANG2\" \
                --save-dir \"$TEXT_DIR/checkpoints\" \
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
                --patience 10 \
            "
    done;
done;
done;
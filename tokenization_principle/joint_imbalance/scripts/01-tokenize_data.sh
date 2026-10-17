#!/usr/bin/bash

function balance_to_num {
    T=$(sed 's/k/000/g' <<<"$1")
    echo $T
}

for BALANCE_NAME in "300k-300k" "500k-100k" "100k-500k" "600k-000k" "000k-600k"; do
    BALANCE=$(balance_to_num $BALANCE_NAME)
    IFS='-' read -r -a BALANCE <<< "${BALANCE}";
    BALANCE1="${BALANCE[0]}"
    BALANCE2="${BALANCE[1]}"

    echo "${BALANCE1} --- ${BALANCE2}"

    DIR="data/tokenizer_bpe/${BALANCE_NAME}/"
    mkdir -p $DIR
    
    sbatch --time=0-1 --ntasks=40 --mem-per-cpu=1G \
    -J "tokenize_${BALANCE_NAME}" \
    --output=logs/%x.out --error=logs/%x.err \
    --wrap="python3 -m tokenization_principle.joint_imbalance.tokenizers_wrap \
        --model bpe \
        --logfile computed/glossology.jsonl \
        --vocab-output $DIR/model.json \
        --process-output $DIR/dev.en $DIR/dev.de $DIR/test.en $DIR/test.de $DIR/train.en $DIR/train.de \
        --vocab-size 4000 \
        --number-of-lines $BALANCE1 $BALANCE2 \
        --model bpe"
done
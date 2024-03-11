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

for FILES in \
    "data/CCrawl.de-en/train.tok.en^data/CCrawl.de-en/train.tok.de" \
    "data/CCrawl.cs-en/train.tok.en^data/CCrawl.cs-en/train.tok.cs" \
    "data/PCrawl.zh-en/train.tok.en^data/PCrawl.zh-en/train.tok.zh"; do

    IFS='^' read -r -a FILES <<< "${FILES}";
    FILE1="${FILES[0]}"
    FILE2="${FILES[1]}"

    LANGS=$(file_to_lang $FILE1)
    # smaller vocab range
    for VOCAB_SIZE in "4000" "8000" "16000" "32000"; do
        # smaller temperature range
        for TEMPERATURE in "0.05" "0.4" "100" "-0.9" "-0.2"; do
        # for TEMPERATURE in "0.05" "0.2" "0.4" "0.9" "100" "-100" "-0.9" "-0.4" "-0.2" "-0.00001"; do
            TEMPERATURE_NAME=$(temperature_name $TEMPERATURE)
            VOCAB_SIZE_NAME=$(vocab_size_name $VOCAB_SIZE)
            SIGNATURE="${TEMPERATURE_NAME}_${VOCAB_SIZE_NAME}"
            echo "Submitting BPE random (${LANGS}/${SIGNATURE})";

            mkdir -p "data/model_bpe_random/${LANGS}/${SIGNATURE}"
            sbatch --time=1-0 --ntasks=10 --mem-per-cpu=3G \
                --output="logs/fit_bpe_random_${LANGS}_${SIGNATURE}.log" \
                --job-name="fit_bpe_random_${LANGS}_${SIGNATURE}" \
                --wrap="python3 ./src/fit_bpe.py \
                    --model random \
                    --randomness-dist softmax \
                    --randomness-temp ${TEMPERATURE} \
                    --vocab-output data/model_bpe_random/${LANGS}/${SIGNATURE}/model.bpe_merges \
                    --vocab-size ${VOCAB_SIZE} \
                    --number-of-lines 10000 \
                    --input $FILE1 $FILE2
                ";
        done;
    done;
done;
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

split_to_nol() {
    case $1 in
        "train")
            echo 1000000
        ;;
        "dev")
            echo 50000
        ;;
        "test")
            echo 50000
        ;;
    esac
}

file_to_lang() {
    LANGS=$(sed 's/data\///g' <<<"$1")
    LANGS=$(sed 's/\/train.tok.en//g' <<<"$LANGS")
    LANGS=$(sed 's/.Crawl\.//g' <<<"$LANGS")
    echo $LANGS
}

prefix_to_lang() {
    LANGS=$(sed 's/.Crawl\.//g' <<<"$1")
    LANGS=$(sed 's/.Prawl\.//g' <<<"$LANGS")
    echo $LANGS
}

# "PCrawl.zh-en" "CCrawl.de-en" 
for PREFIX in "CCrawl.cs-en"; do
    LANGS=$(prefix_to_lang $PREFIX)
    LANGSSTR=$LANGS
    IFS='-' read -r -a LANGS <<< "${LANGS}";

    for SPLIT in "dev" "test" "train"; do
    NOL=$(split_to_nol $SPLIT)
    for LANG in "${LANGS[0]}" "${LANGS[1]}"; do
    # for VOCAB_SIZE in "2000" "4000" "8000" "16000" "32000"; do
    #     for TEMPERATURE in "0.05" "0.2" "0.4" "0.9" "100" "-100" "-0.9" "-0.4" "-0.2" "-0.00001"; do
    # smaller vocab range
    for VOCAB_SIZE in "4000" "8000" "16000" "32000"; do
        # smaller temperature range
        for TEMPERATURE in "0.05" "0.4" "100" "-0.9" "-0.2"; do
            TEMPERATURE_NAME=$(temperature_name $TEMPERATURE)
            VOCAB_SIZE_NAME=$(vocab_size_name $VOCAB_SIZE)
            SIGNATURE="${TEMPERATURE_NAME}_${VOCAB_SIZE_NAME}"
            echo "Submitting BPE random (${LANGSSTR}/${SIGNATURE}/${SPLIT})";

            mkdir -p "data/model_bpe_random/${LANGSSTR}/${SIGNATURE}/"

            sbatch --time=0-4 --ntasks=50 --mem-per-cpu=500M \
                --output="logs_repl/apply_bpe_random_${LANGSSTR}_${SIGNATURE}_${SPLIT}.log" \
                --job-name="apply_bpe_random_${LANGSSTR}_${SIGNATURE}_${SPLIT}" \
                --wrap="python3 ./src/apply_bpe.py \
                    --vocab-input data/model_bpe_random/${LANGSTR}/${SIGNATURE}/model.bpe_merges \
                    --input data/${PREFIX}/${SPLIT}.tok.${LANG} \
                    --output data/model_bpe_random/${LANGSSTR}/${SIGNATURE}/${SPLIT}.${LANG} \
                    --number-of-lines ${NOL} \
                    --logfile computed/tandem.jsonl \
                ";
        done;
    done;
    done;
    done;
done;
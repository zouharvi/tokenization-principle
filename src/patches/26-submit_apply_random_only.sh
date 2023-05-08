#!/usr/bin/bash

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

for FILES in \
    "data/CCrawl.de-en/train.tok.en^data/CCrawl.de-en/train.tok.de" \
    "data/CCrawl.cs-en/train.tok.en^data/CCrawl.cs-en/train.tok.cs" \
    "data/PCrawl.zh-en/train.tok.en^data/PCrawl.zh-en/train.tok.zh"; do

    IFS='^' read -r -a FILES <<< "${FILES}";
    FILE1="${FILES[0]}"
    FILE2="${FILES[1]}"

    LANGS=$(file_to_lang $FILE1)
    LANGSSTR=LANGS
    IFS='-' read -r -a LANGS <<< "${LANGS}";
    for SPLIT in "train" "dev" "test"; do
    NOL=$(split_to_nol $SPLIT)
    for LANG in "${LANGS[0]}" "${LANGS[1]}"; do
    for VOCAB_SIZE in "2000" "4000" "8000" "16000" "32000"; do
        for TEMPERATURE in "0.05" "0.2" "0.4" "0.9" "100" "-100" "-0.9" "-0.4" "-0.2" "-0.00001"; do
            TEMPERATURE_NAME=$(temperature_name $TEMPERATURE)
            VOCAB_SIZE_NAME=$(vocab_size_name $VOCAB_SIZE)
            SIGNATURE="${TEMPERATURE_NAME}_${VOCAB_SIZE_NAME}"
            echo "Submitting BPE random (${LANGSSTR}/${SIGNATURE})";

            sbatch --time=1-0 --ntasks=20 --mem-per-cpu=2G \
                --output="logs/apply_bpe_random_${LANGSSTR}_${SIGNATURE}.log" \
                --job-name="apply_bpe_random_${LANGSSTR}_${SIGNATURE}" \
                --wrap="python3 ./src/apply_bpe.py \
                    --vocab-input data/model_bpe_random/${SIGNATURE}/model.bpe_merges \
                    --input data/CCrawl.de-en/${SPLIT}.tok.${LANG} \
                    --output data/model_bpe_random/${LANGSSTR}/${SIGNATURE}/${SPLIT}.${LANG} \
                    --number-of-lines ${NOL} \
                    --logfile computed/glabrus.jsonl \
                ";
        done;
    done;
    done;
    done;
done;
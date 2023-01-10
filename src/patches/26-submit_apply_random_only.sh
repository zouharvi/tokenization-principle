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

for SPLIT in "train" "dev" "test"; do
NOL=$(split_to_nol $SPLIT)
for LANG in "en" "de"; do
for VOCAB_SIZE in "2000" "4000" "8000" "16000" "32000"; do
    for TEMPERATURE in "0.05" "0.2" "0.4" "0.9" "100" "-100" "-0.9" "-0.4" "-0.2" "-0.00001"; do
        TEMPERATURE_NAME=$(temperature_name $TEMPERATURE)
        VOCAB_SIZE_NAME=$(vocab_size_name $VOCAB_SIZE)
        SIGNATURE="${TEMPERATURE_NAME}_${VOCAB_SIZE_NAME}"
        echo "Submitting BPE random (${SIGNATURE})";

        sbatch --time=1-0 --ntasks=20 --mem-per-cpu=2G \
            --output="logs/apply_bpe_random_${SIGNATURE}.log" \
            --job-name="apply_bpe_random_${SIGNATURE}" \
            --wrap="python3 ./src/apply_bpe.py \
                --vocab-input data/model_bpe_random/${SIGNATURE}/model.bpe_merges \
                --input data/CCrawl.de-en/${SPLIT}.tok.${LANG} \
                --output data/model_bpe_random/${SIGNATURE}/${SPLIT}.${LANG} \
                --number-of-lines ${NOL} \
                --logfile computed/glabrus.jsonl \
            ";
    done;
done;
done;
done;
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

for VOCAB_SIZE in "2000" "4000" "8000" "16000" "32000"; do
    for TEMPERATURE in "0.05" "0.2" "0.4" "0.9" "100" "-100" "-0.9" "-0.4" "-0.2" "-0.00001"; do
        TEMPERATURE_NAME=$(temperature_name $TEMPERATURE)
        VOCAB_SIZE_NAME=$(vocab_size_name $VOCAB_SIZE)
        SIGNATURE="${TEMPERATURE_NAME}_${VOCAB_SIZE_NAME}"
        echo "Submitting BPE random (${SIGNATURE})";

        mkdir -p "data/model_bpe_random/${SIGNATURE}"
        sbatch --time=1-0 --ntasks=20 --mem-per-cpu=2G \
            --output="logs/fit_bpe_random_${SIGNATURE}.log" \
            --job-name="fit_bpe_random_${SIGNATURE}" \
            --wrap="python3 ./src/fit_bpe.py \
                --model random \
                --randomness-dist softmax \
                --randomness-temp ${TEMPERATURE} \
                --vocab-output data/model_bpe_random/${SIGNATURE}/model.bpe_merges \
                --vocab-size ${VOCAB_SIZE} \
                --number-of-lines 10000 \
            ";
    done;
done;
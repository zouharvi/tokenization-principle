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
    MODEL="bpe_random"
    TEMPERATURE_NAME=$(temperature_name $TEMPERATURE)
    VOCAB_SIZE_NAME=$(vocab_size_name $VOCAB_SIZE)
    SIGNATURE="${TEMPERATURE_NAME}_${VOCAB_SIZE_NAME}"
    echo "Submitting BPE random (${SIGNATURE})";

    for LANGS in "de-en"; do
        IFS='-' read -r -a LANGS <<< "${LANGS}";
        LANG1="${LANGS[0]}"
        LANG2="${LANGS[1]}"

        ORIG_DIR="data/model_${MODEL}/${SIGNATURE}";
        TEXT_DIR="data_bin/${LANG1}-${LANG2}/model_${MODEL}/${SIGNATURE}";
        mkdir -p ${TEXT_DIR};

        # for SPLIT in "train" "dev" "test"; do
        # for LANG in "en" "de"; do
        #     sed -i "s/ @@/@@ /g" ${ORIG_DIR}/${SPLIT}.${LANG}
        # done; done;

        sbatch --time=0-1 --ntasks=10 --mem-per-cpu=2G \
            --job-name="preprocess_${MODEL}_${SIGNATURE}.${LANG1}-${LANG2}" \
            --output="logs/preprocess_${MODEL}_${SIGNATURE}.${LANG1}-${LANG2}.log" \
            --wrap="fairseq-preprocess --source-lang $LANG1 --target-lang $LANG2 \
                --trainpref $ORIG_DIR/train --validpref $ORIG_DIR/dev --testpref $ORIG_DIR/test  \
                --destdir $TEXT_DIR \
                --bpe fastbpe \
                --joined-dictionary \
                --tokenizer moses \
                --workers 10 \
            ";
    done;
done;
done;
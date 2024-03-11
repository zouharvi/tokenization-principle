#!/usr/bin/bash

for TRAIN_LINES_NAME in "2k" "8k" "100k"; do
for VOCAB_SIZE_NAME in "4k" "8k" "16k"; do
    for MODEL in \
        "morfessor" "lzw"\
        "tokenizer_unigram" "tokenizer_wordpiece" "tokenizer_bpe"; \
    do
        # for LANGS in "en-de" "de-en"; do
        for LANGS in "de-en"; do
            IFS='-' read -r -a LANGS <<< "${LANGS}";
            LANG1="${LANGS[0]}"
            LANG2="${LANGS[1]}"
            SIGNATURE="l${TRAIN_LINES_NAME}_${VOCAB_SIZE_NAME}"

            echo "SIGNATURE ${MODEL}/${SIGNATURE} ${LANG1}-${LANG2}"

            ORIG_DIR="data/model_${MODEL}/${SIGNATURE}";
            TEXT_DIR="data_bin/${LANG1}-${LANG2}/model_${MODEL}/${SIGNATURE}";
            mkdir -p ${TEXT_DIR};

            # for SPLIT in "train" "dev" "test"; do
            # for LANG in "en" "de"; do
            #     sed -i "s/ @@/@@ /g" ${ORIG_DIR}/${SPLIT}.${LANG}
            # done; done;

            sbatch --time=0-1 --ntasks=20 --mem-per-cpu=2G \
                --job-name="preprocess_${MODEL}_${SIGNATURE}.${LANG1}-${LANG2}" \
                --output="logs/preprocess_${MODEL}_${SIGNATURE}.${LANG1}-${LANG2}.log" \
                --wrap="fairseq-preprocess --source-lang $LANG1 --target-lang $LANG2 \
                    --trainpref $ORIG_DIR/train --validpref $ORIG_DIR/dev --testpref $ORIG_DIR/test  \
                    --destdir $TEXT_DIR \
                    --bpe fastbpe \
                    --joined-dictionary \
                    --tokenizer moses \
                    --workers 20 \
                ";
        done;

    done;
done;
done;

#!/usr/bin/bash

for TRAIN_SIZE_NAME in "0.1" "0.2" "0.4" "0.9" "100"; do
    MODEL="bpe_random"
    for LANGS in "de-en"; do
        IFS='-' read -r -a LANGS <<< "${LANGS}";
        LANG1="${LANGS[0]}"
        LANG2="${LANGS[1]}"

        ORIG_DIR="data/model_${MODEL}/${TRAIN_SIZE_NAME}";
        TEXT_DIR="data_bin/${LANG1}-${LANG2}/model_${MODEL}/${TRAIN_SIZE_NAME}";
        mkdir -p ${TEXT_DIR};


        for SPLIT in "train" "dev" "test"; do
        for LANG in "en" "de"; do
            sed -i "s/ @@/@@ /g" ${ORIG_DIR}/${SPLIT}.${LANG}
        done; done;

        sbatch --time=0-1 --ntasks=10 --mem-per-cpu=2G \
            --job-name="preprocess_${MODEL}_${TRAIN_SIZE_NAME}.${LANG1}-${LANG2}" \
            --output="logs/preprocess_${MODEL}_${TRAIN_SIZE_NAME}.${LANG1}-${LANG2}.log" \
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
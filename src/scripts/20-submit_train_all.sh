#!/usr/bin/bash

vocab_size_name() {
    T=$(sed 's/000$/k/g' <<<"$1")
    echo $T
}

for VOCAB_SIZE in "4000" "8000" "16000"; do
for TRAIN_LINES in "100000" "8000" "2000"; do
    TRAIN_LINES_NAME=$(vocab_size_name $TRAIN_LINES)
    VOCAB_SIZE_NAME=$(vocab_size_name $VOCAB_SIZE)
    SIGNATURE="l${TRAIN_LINES_NAME}_${VOCAB_SIZE_NAME}"
    mkdir -p data/{model_lzw,model_morfessor}/${SIGNATURE}
    echo "SIGNATURE ${SIGNATURE}"

    echo "Submitting LZW";
    sbatch --time=1-0 --ntasks=20 --mem-per-cpu=2G \
        --output="/cluster/home/vzouhar/random-bpe/logs/fit_lzw_${SIGNATURE}.log" \
        --job-name="fit_lzw_${SIGNATURE}" \
        --wrap="python3 ./src/lempel_ziv_welsch/fit_lzw.py \
            --vocab-output data/model_lzw/${SIGNATURE}/model.vocab \
            --vocab-size ${VOCAB_SIZE} \
            --number-of-lines ${TRAIN_LINES} \
        ";

    echo "Submitting Morfessor";
    sbatch --time=1-0 --ntasks=20 --mem-per-cpu=2G \
        --output="/cluster/home/vzouhar/random-bpe/logs/fit_morfessor_${SIGNATURE}.log" \
        --job-name="fit_morfessor_${SIGNATURE}" \
        --wrap="python3 ./src/wrappers/morfessor_wrap_train.py \
            --vocab-output data/model_morfessor/${SIGNATURE}/model.pkl \
            --vocab-size ${VOCAB_SIZE} \
            --number-of-lines ${TRAIN_LINES} \
        ";

    for TOKENIZER_METHOD in "bpe" "unigram" "wordpiece"; do
        mkdir -p data/model_tokenizer_${TOKENIZER_METHOD}/${SIGNATURE}

        echo "Submitting tokenizer/${TOKENIZER_METHOD} (${TRAIN_LINES_NAME})";
        sbatch --time=1-0 --ntasks=20 --mem-per-cpu=2G \
            --output="/cluster/home/vzouhar/random-bpe/logs/fit_tokenizer_${TOKENIZER_METHOD}_${SIGNATURE}.log" \
            --job-name="fit_tokenizer_${TOKENIZER_METHOD}_${SIGNATURE}" \
            --wrap="python3 ./src/wrappers/tokenizers_wrap.py \
                --vocab-output data/model_tokenizer_${TOKENIZER_METHOD}/${SIGNATURE}/model.json \
                --model ${TOKENIZER_METHOD} \
                --vocab-size ${VOCAB_SIZE} \
                --number-of-lines ${TRAIN_LINES} \
                --logfile computed/paten.jsonl \
                --process-output \
                        data/model_tokenizer_${TOKENIZER_METHOD}/${SIGNATURE}/dev.en \
                        data/model_tokenizer_${TOKENIZER_METHOD}/${SIGNATURE}/dev.de \
                        data/model_tokenizer_${TOKENIZER_METHOD}/${SIGNATURE}/test.en \
                        data/model_tokenizer_${TOKENIZER_METHOD}/${SIGNATURE}/test.de \
                        data/model_tokenizer_${TOKENIZER_METHOD}/${SIGNATURE}/train.en \
                        data/model_tokenizer_${TOKENIZER_METHOD}/${SIGNATURE}/train.de \
            ";
        done
done;
done;
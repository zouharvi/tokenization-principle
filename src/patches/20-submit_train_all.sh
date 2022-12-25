#!/usr/bin/bash

VOCAB_SIZE="8192"

train_lines_name() {
    case $1 in
        "400000")
            echo "400k"
        ;;
        "100000")
            echo "100k"
        ;;
        "10000")
            echo "10k"
        ;;
        "5000")
            echo "5k"
        ;;
        "1000")
            echo "1k"
        ;;
    esac
}

for TRAIN_LINES in "400000" "100000" "25000" "5000"; do
    TRAIN_LINES_NAME=$(train_lines_name $TRAIN_LINES)
    SIGNATURE="obturate_${VOCAB_SIZE}_${TRAIN_LINES_NAME}"
    mkdir -p data/{model_bpe_greedy,model_bpe_antigreedy,model_bpe_captrick,model_lzw,model_morfessor}/${TRAIN_LINES_NAME}

    echo "Submitting BPE (greedy)";
    sbatch --time=1-0 --ntasks=20 --mem-per-cpu=2G \
        --output="logs/fit_bpe_greedy_${SIGNATURE}.log" \
        --job-name="fit_bpe_greedy_${SIGNATURE}" \
        --wrap="python3 ./src/fit_bpe.py \
            --model greedy \
            --vocab-output data/model_bpe_greedy/${TRAIN_LINES_NAME}/model.bpe_merges \
            --vocab-size ${VOCAB_SIZE} \
            --number-of-lines ${TRAIN_LINES} \
        ";

    echo "Submitting BPE (antigreedy)";
    sbatch --time=1-0 --ntasks=20 --mem-per-cpu=2G \
        --output="logs/fit_bpe_antigreedy_${SIGNATURE}.log" \
        --job-name="fit_bpe_antigreedy_${SIGNATURE}" \
        --wrap="python3 ./src/fit_bpe.py \
            --model antigreedyalmost \
            --threshold 2 \
            --vocab-output data/model_bpe_antigreedy/${TRAIN_LINES_NAME}/model.bpe_merges \
            --vocab-size ${VOCAB_SIZE} \
            --number-of-lines ${TRAIN_LINES} \
        ";

    echo "Submitting BPE (captrick)";
    sbatch --time=1-0 --ntasks=20 --mem-per-cpu=2G \
        --output="logs/fit_bpe_greedycaptrick_${SIGNATURE}.log" \
        --job-name="fit_bpe_greedycaptrick_${SIGNATURE}" \
        --wrap="python3 ./src/fit_bpe.py \
            --model greedycaptrick \
            --vocab-output data/model_bpe_captrick/${TRAIN_LINES_NAME}/model.bpe_merges \
            --vocab-size ${VOCAB_SIZE} \
            --number-of-lines ${TRAIN_LINES} \
        ";

    echo "Submitting LZW";
    sbatch --time=1-0 --ntasks=20 --mem-per-cpu=2G \
        --output="logs/fit_lzw_${SIGNATURE}.log" \
        --job-name="fit_lzw_${SIGNATURE}" \
        --wrap="python3 ./src/lempel_ziv_welsch/fit_lzw.py \
            --vocab-output data/model_lzw/${TRAIN_LINES_NAME}/model.vocab \
            --vocab-size ${VOCAB_SIZE} \
            --number-of-lines ${TRAIN_LINES} \
        ";

    echo "Submitting Morfessor";
    sbatch --time=1-0 --ntasks=20 --mem-per-cpu=2G \
        --output="logs/fit_morfessor_${SIGNATURE}.log" \
        --job-name="fit_morfessor_${SIGNATURE}" \
        --wrap="python3 ./src/wrappers/morfessor_wrap_train.py \
            --vocab-output data/model_morfessor/${TRAIN_LINES_NAME}/model.pkl \
            --vocab-size ${VOCAB_SIZE} \
            --number-of-lines ${TRAIN_LINES} \
        ";

    for TOKENIZER_METHOD in "bpe" "unigram" "wordpiece"; do
        mkdir -p data/model_tokenizer_${TOKENIZER_METHOD}/${TRAIN_LINES_NAME}

        echo "Submitting tokenizer/${TOKENIZER_METHOD} (${TRAIN_LINES_NAME})";
        sbatch --time=1-0 --ntasks=20 --mem-per-cpu=2G \
            --output="logs/fit_tokenizer_${TOKENIZER_METHOD}_${SIGNATURE}.log" \
            --job-name="fit_tokenizer_${TOKENIZER_METHOD}_${SIGNATURE}" \
            --wrap="python3 ./src/wrappers/tokenizers_wrap.py \
                --vocab-output data/model_tokenizer_${TOKENIZER_METHOD}/${TRAIN_LINES_NAME}/model.json \
                --model ${TOKENIZER_METHOD} \
                --vocab-size ${VOCAB_SIZE} \
                --number-of-lines ${TRAIN_LINES} \
                --logfile computed/azaroth.jsonl \
                --process-output \
                        data/model_tokenizer_${TOKENIZER_METHOD}/${TRAIN_LINES_NAME}/dev.en \
                        data/model_tokenizer_${TOKENIZER_METHOD}/${TRAIN_LINES_NAME}/dev.de \
                        data/model_tokenizer_${TOKENIZER_METHOD}/${TRAIN_LINES_NAME}/test.en \
                        data/model_tokenizer_${TOKENIZER_METHOD}/${TRAIN_LINES_NAME}/test.de \
                        data/model_tokenizer_${TOKENIZER_METHOD}/${TRAIN_LINES_NAME}/train.en \
                        data/model_tokenizer_${TOKENIZER_METHOD}/${TRAIN_LINES_NAME}/train.de \
            ";
        done
done

# echo "Submitting SentencePiece (also for applying)";
# sbatch --time=1-0 --ntasks=20 --mem-per-cpu=2G \
#     --output="logs/fit_spm_${SIGNATURE}.log" \
#     --job-name="fit_spm_${SIGNATURE}" \
#     --wrap="python3 ./src/wrappers/sentencepiece_wrap.py \
#         --vocab-output data/model_spm/ \
#         --vocab-size ${VOCAB_SIZE} \
#         --number-of-lines ${TRAIN_LINES} \
#     ";

python3 ./src/lempel_ziv_welsch/fit_lzw.py \
    --vocab-output data/model_lzw/model.vocab \
    --vocab-size 8192 \
    --number-of-lines 1000

python3 ./src/lempel_ziv_welsch/apply_lzw.py \
    --input data/CCrawl.de-en/dev.tok.en \
    --output data/model_lzw/tmp.dev.en \
    --vocab-input data/model_lzw/model.vocab \
    --number-of-lines 100
# TODO: test that applying resolves UNK well

python3 ./src/apply_bpe.py \
    --vocab-input data/model_bpe_captrick/model.bpe_merges \
    --input data/CCrawl.de-en/dev.tok.en \
    --output data/model_bpe_captrick/dev.en \
    --number-of-lines 100 \
    --captrickflag
#!/usr/bin/bash

VOCAB_SIZE="8192"
TRAIN_LINES="100000"
SIGNATURE="obturate_${VOCAB_SIZE}_${TRAIN_LINES}"
mkdir -p data/{model_bpe_greedy,model_bpe_captrick,model_spm,model_lzw,model_morfessor}

echo "Submitting BPE";
sbatch --time=1-0 --ntasks=20 --mem-per-cpu=2G \
    --output="logs/fit_bpe_greedy_${SIGNATURE}.log" \
    --job-name="fit_bpe_greedy_${SIGNATURE}" \
    --wrap="python3 ./src/fit_bpe.py \
        --model greedy \
        --vocab-output data/model_bpe_greedy/model.bpe_merges \
        --vocab-size ${VOCAB_SIZE} \
        --number-of-lines ${TRAIN_LINES} \
    ";

echo "Submitting BPE";
sbatch --time=1-0 --ntasks=20 --mem-per-cpu=2G \
    --output="logs/fit_bpe_greedycaptrick_${SIGNATURE}.log" \
    --job-name="fit_bpe_greedycaptrick_${SIGNATURE}" \
    --wrap="python3 ./src/fit_bpe.py \
        --model greedycaptrick \
        --vocab-output data/model_bpe_captrick/model.bpe_merges \
        --vocab-size ${VOCAB_SIZE} \
        --number-of-lines ${TRAIN_LINES} \
    ";

echo "Submitting SentencePiece (also for applying)";
sbatch --time=1-0 --ntasks=20 --mem-per-cpu=2G \
    --output="logs/fit_spm_${SIGNATURE}.log" \
    --job-name="fit_spm_${SIGNATURE}" \
    --wrap="python3 ./src/wrappers/sentencepiece_wrap.py \
        --vocab-output data/model_spm/ \
        --vocab-size ${VOCAB_SIZE} \
        --number-of-lines ${TRAIN_LINES} \
    ";

echo "Submitting LZW";
sbatch --time=1-0 --ntasks=20 --mem-per-cpu=2G \
    --output="logs/fit_lzw_${SIGNATURE}.log" \
    --job-name="fit_lzw_${SIGNATURE}" \
    --wrap="python3 ./src/lempel_ziv_welsch/fit_lzw.py \
        --vocab-output data/model_lzw/model.vocab \
        --vocab-size ${VOCAB_SIZE} \
        --number-of-lines ${TRAIN_LINES} \
    ";

echo "Submitting Morfessor";
sbatch --time=1-0 --ntasks=20 --mem-per-cpu=2G \
    --output="logs/fit_morfessor_${SIGNATURE}.log" \
    --job-name="fit_morfessor_${SIGNATURE}" \
    --wrap="python3 ./src/wrappers/morfessor_wrap_train.py \
        --vocab-output data/model_morfessor/model.pkl \
        --vocab-size ${VOCAB_SIZE} \
        --number-of-lines ${TRAIN_LINES} \
    ";
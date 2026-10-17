#!/usr/bin/bash

# Unified Tokenization Submission Script
# Fits and applies all tokenizers (Unigram, WordPiece, LZW, Morfessor, BPE variants)
# across specified language pairs and vocabulary sizes into data/prepared/<lang>/<signature>/

LANGS_LIST=("de-en" "cs-en" "zh-en")
VOCAB_SIZES=("4000" "8000" "16000" "32000")

mkdir -p logs

for LANGS in "${LANGS_LIST[@]}"; do
    IFS='-' read -r -a L <<< "${LANGS}"
    DATA_DIR="data/${LANGS}"

    for VOCAB_SIZE in "${VOCAB_SIZES[@]}"; do
        # 1. Standard Tokenizers (Unigram, WordPiece, LZW, Morfessor, Greedy BPE)
        for TOKENIZER in "unigram" "wordpiece" "lzw" "morfessor" "greedy_bpe"; do
            SIGNATURE="${TOKENIZER}_v${VOCAB_SIZE}"
            OUT_DIR="data/prepared/${LANGS}/${SIGNATURE}"
            mkdir -p "${OUT_DIR}"

            echo "Submitting ${LANGS} ${SIGNATURE}"
            sbatch --time=04:00:00 --ntasks=10 --mem-per-cpu=2G \
                -J "prep_${LANGS}_${SIGNATURE}" \
                --output="logs/prep_${LANGS}_${SIGNATURE}.log" \
                --wrap="python3 -m tokenization_principle.prepare_data \
                    --tokenizer ${TOKENIZER} \
                    --vocab-size ${VOCAB_SIZE} \
                    --lang-pair ${LANGS} \
                    --input-dir ${DATA_DIR} \
                    --output-dir ${OUT_DIR} \
                "
        done

        # 2. Randomized BPE across temperatures
        for TEMP in "0.05" "0.4" "100" "-0.9" "-0.2"; do
            T_NAME=$(sed 's/^-/m/; s/^0./0/; s/m0./m0/' <<< "$TEMP")
            SIGNATURE="random_bpe_t${T_NAME}_v${VOCAB_SIZE}"
            OUT_DIR="data/prepared/${LANGS}/${SIGNATURE}"
            mkdir -p "${OUT_DIR}"

            echo "Submitting ${LANGS} ${SIGNATURE}"
            sbatch --time=04:00:00 --ntasks=10 --mem-per-cpu=2G \
                -J "prep_${LANGS}_${SIGNATURE}" \
                --output="logs/prep_${LANGS}_${SIGNATURE}.log" \
                --wrap="python3 -m tokenization_principle.prepare_data \
                    --tokenizer random_bpe \
                    --temperature ${TEMP} \
                    --vocab-size ${VOCAB_SIZE} \
                    --lang-pair ${LANGS} \
                    --input-dir ${DATA_DIR} \
                    --output-dir ${OUT_DIR} \
                "
        done
    done
done
#!/usr/bin/bash

# /src/apply_bpe.py --vocab-input data/greedy_pmi/1000/model.bpe --input data/CCrawl.de-en/dev.tok.en --output data/greedy_pmi/1000/dev.en --number-of-lines 50000 --logfile computed/neophone_tmp.jsonl

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

for VOCAB_SIZE in "2000" "4000"; do
    for BPE_MODEL in "greedy_pmi" "greedy" "random_pmi_freq"; do
        for SPLIT in "train" "dev"; do
            NO_LINES=$(split_to_nol $SPLIT)

            for LANG in "en" "de"; do
                SIGNATURE="${SPLIT}_${LANG}_${BPE_MODEL}_${VOCAB_SIZE}"
                BPE_PATH="data/${BPE_MODEL}/${VOCAB_SIZE}/model.bpe"
                SRC_DATA_PATH="data/CCrawl.de-en/${SPLIT}.tok.${LANG}"
                TGT_DATA_PATH="data/${BPE_MODEL}/${VOCAB_SIZE}/${SPLIT}.${LANG}"

                echo "Submitting ${SIGNATURE}";
                sbatch --time=1-0 --ntasks=50 --mem-per-cpu=1G \
                    --output="logs/apply_bpe_${SIGNATURE}.log" \
                    --job-name="apply_bpe_${SIGNATURE}" \
                    --wrap="python3 ./src/apply_bpe.py \
                        --input ${SRC_DATA_PATH} \
                        --output ${TGT_DATA_PATH} \
                        --vocab-input ${BPE_PATH} \
                        --number-of-lines ${NO_LINES} \
                        --method merge_operations \
                        --logfile computed/neophone.jsonl \
                    ";
            done;
        done;
    done;
done;

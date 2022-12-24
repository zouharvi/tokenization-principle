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

for TRAIN_LINES in "100000"; do
# for TRAIN_LINES in "5000" "25000" "100000" "400000"; do
    TRAIN_LINES_NAME=$(train_lines_name $TRAIN_LINES)
    for SPLIT in "train" "dev" "test"; do
    for LANG in "en" "de"; do
        NOL=$(split_to_nol $SPLIT)
        SIGNATURE="azaroth_${SPLIT}_${LANG}_${TRAIN_LINES_NAME}"

        echo "Submitting LZW";
        sbatch --time=1-0 --ntasks=20 --mem-per-cpu=2G \
            --output="logs/apply_lzw_${SIGNATURE}.log" \
            --job-name="apply_lzw_${SIGNATURE}" \
            --wrap="python3 ./src/lempel_ziv_welsch/apply_lzw.py \
                --vocab-input data/model_lzw/${TRAIN_LINES_NAME}/model.vocab \
                --input data/CCrawl.de-en/${SPLIT}.tok.${LANG} \
                --output data/model_lzw/${TRAIN_LINES_NAME}/${SPLIT}.${LANG} \
                --number-of-lines ${NOL} \
                --logfile computed/azaroth.jsonl \
            ";

        echo "Submitting BPE greedy";
        sbatch --time=1-0 --ntasks=20 --mem-per-cpu=2G \
            --output="logs/apply_bpe_greedy_${SIGNATURE}.log" \
            --job-name="apply_bpe_greedy_${SIGNATURE}" \
            --wrap="python3 ./src/apply_bpe.py \
                --vocab-input data/model_bpe_greedy/${TRAIN_LINES_NAME}/model.bpe_merges \
                --input data/CCrawl.de-en/${SPLIT}.tok.${LANG} \
                --output data/model_bpe_greedy/${TRAIN_LINES_NAME}/${SPLIT}.${LANG} \
                --number-of-lines ${NOL} \
                --logfile computed/azaroth.jsonl \
            ";

        echo "Submitting BPE Captrick";
        sbatch --time=1-0 --ntasks=20 --mem-per-cpu=2G \
            --output="logs/apply_bpe_captrick_${SIGNATURE}.log" \
            --job-name="apply_bpe_captrick_${SIGNATURE}" \
            --wrap="python3 ./src/apply_bpe.py \
                --vocab-input data/model_bpe_captrick/${TRAIN_LINES_NAME}/model.bpe_merges \
                --input data/CCrawl.de-en/${SPLIT}.tok.${LANG} \
                --output data/model_bpe_captrick/${TRAIN_LINES_NAME}/${SPLIT}.${LANG} \
                --number-of-lines ${NOL} \
                --logfile computed/azaroth.jsonl \
                --captrickflag \
            ";

        echo "Submitting BPE Antigreedy";
        sbatch --time=1-0 --ntasks=20 --mem-per-cpu=2G \
            --output="logs/apply_bpe_antigreedy_${SIGNATURE}.log" \
            --job-name="apply_bpe_antigreedy_${SIGNATURE}" \
            --wrap="python3 ./src/apply_bpe.py \
                --vocab-input data/model_bpe_antigreedy/${TRAIN_LINES_NAME}/model.bpe_merges \
                --input data/CCrawl.de-en/${SPLIT}.tok.${LANG} \
                --output data/model_bpe_antigreedy/${TRAIN_LINES_NAME}/${SPLIT}.${LANG} \
                --number-of-lines ${NOL} \
                --logfile computed/azaroth.jsonl \
            ";

        echo "Submitting Morfessor";
        sbatch --time=1-0 --ntasks=20 --mem-per-cpu=2G \
            --output="logs/apply_morfessor_${SIGNATURE}.log" \
            --job-name="apply_morfessor_${SIGNATURE}" \
            --wrap="python3 ./src/wrappers/morfessor_wrap_apply.py \
                --model data/model_morfessor/${TRAIN_LINES_NAME}/model.pkl \
                --input data/CCrawl.de-en/${SPLIT}.tok.${LANG} \
                --output data/model_morfessor/${TRAIN_LINES_NAME}/${SPLIT}.${LANG} \
                --number-of-lines ${NOL} \
                --logfile computed/azaroth.jsonl \
            ";
    done;
    done;
done;
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

vocab_size_name() {
    T=$(sed 's/000$/k/g' <<<"$1")
    echo $T
}


for SPLIT in "train" "dev" "test"; do
NOL=$(split_to_nol $SPLIT)
for LANG in "en" "de"; do
    for VOCAB_SIZE in "4000" "8000" "16000"; do
    for TRAIN_LINES in "2000" "8000" "100000"; do
        TRAIN_LINES_NAME=$(vocab_size_name $TRAIN_LINES)
        VOCAB_SIZE_NAME=$(vocab_size_name $VOCAB_SIZE)
        SIGNATURE="l${TRAIN_LINES_NAME}_${VOCAB_SIZE_NAME}"

        echo "Submitting LZW";
        sbatch --time=1-0 --ntasks=20 --mem-per-cpu=2G \
            --output="logs/apply_lzw_${LANG}_${SPLIT}_${SIGNATURE}.log" \
            --job-name="apply_lzw_${LANG}_${SPLIT}_${SIGNATURE}" \
            --wrap="python3 ./src/lempel_ziv_welsch/apply_lzw.py \
                --vocab-input data/model_lzw/${SIGNATURE}/model.vocab \
                --input data/CCrawl.de-en/${SPLIT}.tok.${LANG} \
                --output data/model_lzw/${SIGNATURE}/${SPLIT}.${LANG} \
                --number-of-lines ${NOL} \
                --logfile computed/paten.jsonl \
            ";

        echo "Submitting Morfessor";
        sbatch --time=1-0 --ntasks=20 --mem-per-cpu=2G \
            --output="logs/apply_morfessor_${LANG}_${SPLIT}_${SIGNATURE}.log" \
            --job-name="apply_morfessor_${LANG}_${SPLIT}_${SIGNATURE}" \
            --wrap="python3 ./src/wrappers/morfessor_wrap_apply.py \
                --model data/model_morfessor/${SIGNATURE}/model.pkl \
                --input data/CCrawl.de-en/${SPLIT}.tok.${LANG} \
                --output data/model_morfessor/${SIGNATURE}/${SPLIT}.${LANG} \
                --number-of-lines ${NOL} \
                --logfile computed/paten.jsonl \
            ";
    done;

    # for TEMPERATURE in "0.4" "1000"; do
    #     echo "Submitting BPE Random";
    #     sbatch --time=1-0 --ntasks=20 --mem-per-cpu=2G \
    #         --output="logs/apply_bpe_random_${SIGNATURE}.log" \
    #         --job-name="apply_bpe_random_${SIGNATURE}" \
    #         --wrap="python3 ./src/apply_bpe.py \
    #             --vocab-input data/model_bpe_random/${TEMPERATURE}/model.bpe_merges \
    #             --input data/CCrawl.de-en/${SPLIT}.tok.${LANG} \
    #             --output data/model_bpe_random/${TEMPERATURE}/${SPLIT}.${LANG} \
    #             --number-of-lines ${NOL} \
    #             --logfile computed/bouree.jsonl \
    #         ";
    # done;
done;
done;
done;
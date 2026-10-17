#!/usr/bin/bash

for LANGS in "de-en" "cs-en" "zh-en"; do
    IFS='-' read -r -a L <<< "${LANGS}"
    for SPLIT in "train" "dev" "test"; do
        for LANG in "${L[0]}" "${L[1]}"; do
            sbatch --time=0-4 --ntasks=20 --mem-per-cpu=1G \
                -J "tokenize_${LANGS}.${SPLIT}.${LANG}" \
                --output=logs/%x.out --error=logs/%x.err \
                --wrap="cat \"data/${LANGS}/${SPLIT}.${LANG}\" \
                    | sacremoses -j 20 -l ${LANG} tokenize \
                    > \"data/${LANGS}/${SPLIT}.tok.${LANG}\"\
                "
        done
    done
done
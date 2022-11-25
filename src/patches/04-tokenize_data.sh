#!/usr/bin/bash


for SPLIT in "train" "dev" "test"; do
    for LANG in "en" "de"; do
            sbatch --time=0-4 --ntasks=40 --mem-per-cpu=1G \
                --output="logs/tokenize_${SPLIT}.${LANG}.log" \
                --job-name="tokenize_${SPLIT}.${LANG}" \
                --wrap="cat \"data/CCrawl.de-en/${SPLIT}.${LANG}\" \
                    | sacremoses -j 40 -l ${LANG} tokenize \
                    > \"data/CCrawl.de-en/${SPLIT}.tok.${LANG}\"\
                ";
    done;
done;
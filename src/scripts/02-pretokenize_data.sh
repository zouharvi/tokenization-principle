#!/usr/bin/bash

prefix_to_lang() {
    LANGS=$(sed 's/.Crawl\.//g' <<<"$1")
    echo $LANGS
}

for PREFIX in "CCrawl.de-en" "CCrawl.cs-en" "PCrawl.zh-en"; do
for SPLIT in "train" "dev" "test"; do
    LANGS=$(prefix_to_lang $PREFIX)
    IFS='-' read -r -a LANGS <<< "${LANGS}";
    for LANG in "${LANGS[0]}" "${LANGS[1]}"; do
        sbatch --time=0-4 --ntasks=5 --mem-per-cpu=10G \
            --output="logs/tokenize_${PREFIX}.${SPLIT}.${LANG}.log" \
            --job-name="tokenize_${PREFIX}.${SPLIT}.${LANG}" \
            --wrap="cat \"data/${PREFIX}/${SPLIT}.${LANG}\" \
                | sacremoses -j 20 -l ${LANG} tokenize \
                > \"data/${PREFIX}/${SPLIT}.tok.${LANG}\"\
            ";
    done;
done;
done;

PREFIX="CCrawl.de-en"
SPLIT="train"
LANG="en"
sbatch --time=0-4 --ntasks=5 --mem-per-cpu=10G \
--output="logs/tokenize_${PREFIX}.${SPLIT}.${LANG}.log" \
--job-name="tokenize_${PREFIX}.${SPLIT}.${LANG}" \
--wrap="cat \"data/${PREFIX}/${SPLIT}.${LANG}\" \
    | sacremoses -j 20 -l ${LANG} tokenize \
    > \"data/${PREFIX}/${SPLIT}.tok.${LANG}\"\
";
done;
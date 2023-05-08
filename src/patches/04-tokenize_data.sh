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
        sbatch --time=0-4 --ntasks=20 --mem-per-cpu=1G \
            --output="logs/tokenize_${PREFIX}.${SPLIT}.${LANG}.log" \
            --job-name="tokenize_${PREFIX}.${SPLIT}.${LANG}" \
            --wrap="cat \"data/${PREFIX}/${SPLIT}.${LANG}\" \
                | sacremoses -j 20 -l ${LANG} tokenize \
                > \"data/${PREFIX}/${SPLIT}.tok.${LANG}\"\
            ";
    done;
done;
done;

# SPLIT="train"
# LANG="de"
# cat "data/CCrawl.de-en/${SPLIT}.${LANG}" \
#     | sacremoses -j 22 -l ${LANG} tokenize \
#     > "data/CCrawl.de-en/${SPLIT}.tok.${LANG}"
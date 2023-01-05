#!/usr/bin/bash

rm -f computed/freq_prob_renyi_alphas_grid.jsonl

for ALPHA in $(seq 0.1 0.05 5.00); do
    # run without graphical
    OUTPUT=$(DISPLAY="" ./src/patches/30-bleu_fig_recompute.py --predictor freq_prob_renyi --freq-alpha-start 0.25 --freq-alpha-end 0.5 --renyi-alpha $ALPHA --use-cache | grep "JSON!")
    OUTPUT=${OUTPUT#"JSON!"}
    echo $ALPHA $OUTPUT
    echo $OUTPUT >> computed/freq_prob_renyi_alphas_grid.jsonl
done;
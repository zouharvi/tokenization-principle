#!/usr/bin/bash

rm -f computed/freq_prob_renyi_alphas_grid_2.jsonl

for ALPHA in $(seq 0.1 0.05 5.00); do
    # run without graphical
    OUTPUT=$(DISPLAY="" ./src/patches/30-bleu_fig_recompute.py --predictor freq_prob_renyi --freq-alpha-start 0.0 --freq-alpha-end 1.0 --renyi-alpha $ALPHA --use-cache | grep "JSON!")
    OUTPUT=${OUTPUT#"JSON!"}
    echo $ALPHA $OUTPUT
    echo $OUTPUT >> computed/freq_prob_renyi_alphas_grid_2.jsonl
done;
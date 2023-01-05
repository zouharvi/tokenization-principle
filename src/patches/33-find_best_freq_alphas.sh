#!/usr/bin/bash

rm -f computed/freq_alphas_grid.jsonl

for START_A in $(seq 0.0 0.05 1.00); do
for END_A in $(seq 0.0 0.05 1.00); do
    if (( $(echo "$START_A > $END_A" |bc -l) )); then
        continue
    fi
    # run without graphical
    OUTPUT=$(DISPLAY="" ./src/patches/30-bleu_fig_recompute.py --predictor freq --freq-alpha-start $START_A --freq-alpha-end $END_A --use-cache | grep "JSON!")
    OUTPUT=${OUTPUT#"JSON!"}
    echo $START_A $END_A $OUTPUT
    echo $OUTPUT >> computed/freq_alphas_grid.jsonl
done;
done;
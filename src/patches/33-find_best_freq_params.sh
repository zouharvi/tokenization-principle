#!/usr/bin/bash

# REPLACED BY 39-predict_bleu_freq_prob_fast.py

rm -f computed/freq_prob_alphas_grid.jsonl

for START_A in $(seq 0.0 0.02 1.00); do
for END_A in $(seq 0.0 0.02 1.00); do
    if (( $(echo "$START_A > $END_A" |bc -l) )); then
        continue
    fi
    # run without graphical
    OUTPUT=$(DISPLAY="" ./src/figures/predict_bleu.py --predictor freq_prob --freq-alpha-start $START_A --freq-alpha-end $END_A --power 1 --load-cache | grep "JSON!")
    OUTPUT=${OUTPUT#"JSON!"}
    echo $START_A $END_A $OUTPUT
    echo $OUTPUT >> computed/freq_prob_alphas_grid.jsonl
done;
done;
#!/usr/bin/bash

rm -f computed/renyi_vals_sub.jsonl

function run_prediction {
    # run without graphical output
    OUTPUT=$(./src/figures/predict_bleu.py --load-cache $1 --data-train --no-graphics | grep "JSON!")
    OUTPUT=${OUTPUT#"JSON!"}
    echo $1 $OUTPUT
    echo $OUTPUT >> computed/renyi_vals_sub.jsonl
}

for POWER in $(seq 0.0 0.1 10.00); do
    # run_prediction "--predictor renyi --freq-alpha-start 0.0 --freq-alpha-end 1.0 --power $POWER"
    run_prediction "--predictor renyi_eff --freq-alpha-start 0.0 --freq-alpha-end 1.0 --power $POWER"
done;

run_prediction "--predictor seq_len"
run_prediction "--predictor bits"
run_prediction "--predictor entropy"
run_prediction "--predictor entropy_eff"
#!/usr/bin/bash

rm -f computed/all_predictors_grid.jsonl

function run_prediction {
    # run without graphical output
    OUTPUT=$(DISPLAY="" ./src/figures/predict_bleu.py --no-graphics --load-cache $1 | grep "JSON!")
    OUTPUT=${OUTPUT#"JSON!"}
    echo $OUTPUT
    echo $OUTPUT >> computed/all_predictors_grid.jsonl
}

run_prediction "--predictor subwords"
run_prediction "--predictor seq_len"
run_prediction "--predictor bits"

for START_A in $(seq 0.0 0.1 1.00); do
for END_A in $(seq 0.0 0.1 1.00); do
    if (( $(echo "$START_A > $END_A" |bc -l) )); then
        continue
    fi
    run_prediction "--predictor freq --freq-alpha-start $START_A --freq-alpha-end $END_A"
done;
done;

for START_A in $(seq 0.0 0.1 1.00); do
for END_A in $(seq 0.0 0.1 1.00); do
    if (( $(echo "$START_A > $END_A" |bc -l) )); then
        continue
    fi
    run_prediction "--predictor freq_prob --freq-alpha-start $START_A --freq-alpha-end $END_A"
done;
done;

for POWER in $(seq 0.10 0.10 4.00); do
    run_prediction "--predictor renyi --freq-alpha-start 0.0 --freq-alpha-end 1.0 --power $POWER"
    run_prediction "--predictor renyi --freq-alpha-start 0.15 --freq-alpha-end 0.85 --power $POWER"
    run_prediction "--predictor renyi --freq-alpha-start 0.25 --freq-alpha-end 0.75 --power $POWER"
done;

for POWER in "-1" "0.5" "1.0" "1.5" "2"; do
for CENTRAL_MEASURE in "mean" "median" "mode"; do
for AGGREGATOR in "sqrt" "mean" "median" "sqrt"; do
    run_prediction "--predictor agg_deviation --central-measure ${CENTRAL_MEASURE} --aggregator ${AGGREGATOR} --power $POWER"
done;
done;
done;


run_prediction "--predictor inter_quantile_range --base-vals freqs --freq-alpha-start 0.25 --freq-alpha-end 0.75"
run_prediction "--predictor inter_quantile_range --base-vals probs --freq-alpha-start 0.25 --freq-alpha-end 0.75"
run_prediction "--predictor inter_quantile_range --base-vals freqs --freq-alpha-start 0.15 --freq-alpha-end 0.85"
run_prediction "--predictor inter_quantile_range --base-vals probs --freq-alpha-start 0.15 --freq-alpha-end 0.85"

run_prediction "--predictor entropy"
run_prediction "--predictor coefficient_variation"

run_prediction "--predictor quartile_dispersion --freq-alpha-start 0.25 --freq-alpha-end 0.75"
run_prediction "--predictor quartile_dispersion --freq-alpha-start 0.15 --freq-alpha-end 0.85"
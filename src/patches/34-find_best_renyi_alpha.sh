#!/usr/bin/bash

# rm -f computed/renyi_alphas_grid.jsonl

for ALPHA in $(seq 5.05 0.20 15.00); do
    # run without graphical
    OUTPUT=$(DISPLAY="" ./src/figures/predict_bleu.py --predictor renyi --freq-alpha-start 0.0 --freq-alpha-end 1.0 --renyi-alpha $ALPHA --load-cache | grep "JSON!")
    OUTPUT=${OUTPUT#"JSON!"}
    echo $ALPHA $OUTPUT
    echo $OUTPUT >> computed/renyi_alphas_grid.jsonl
done;


# OUTPUT=$(DISPLAY="" ./src/figures/predict_bleu.py --predictor bits --load-cache | grep "JSON!")
# OUTPUT=${OUTPUT#"JSON!"}
# echo "bits" $OUTPUT
# echo $OUTPUT >> computed/renyi_alphas_grid.jsonl

# OUTPUT=$(DISPLAY="" ./src/figures/predict_bleu.py --predictor seq_len --load-cache | grep "JSON!")
# OUTPUT=${OUTPUT#"JSON!"}
# echo "seq_len" $OUTPUT
# echo $OUTPUT >> computed/renyi_alphas_grid.jsonl

# OUTPUT=$(DISPLAY="" ./src/figures/predict_bleu.py --predictor freq --freq-alpha-start 0.65 --freq-alpha-end 0.75 --load-cache | grep "JSON!")
# OUTPUT=${OUTPUT#"JSON!"}
# echo "freq" $OUTPUT
# echo $OUTPUT >> computed/renyi_alphas_grid.jsonl
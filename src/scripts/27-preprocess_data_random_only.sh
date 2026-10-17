#!/usr/bin/bash

# NOTE: Preprocessing with fairseq-preprocess is no longer needed.
# The PyTorch training pipeline reads the tokenized text files in data/ directly.
echo "No fairseq-preprocess needed for PyTorch training pipeline. Data in data/ is ready to be used directly."
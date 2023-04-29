#!/usr/bin/env python3

import argparse
from bpe_models import get_model

args = argparse.ArgumentParser()
args.add_argument("-i", "--input", nargs="+", default=[
    "data/CCrawl.de-en/train.tok.en", "data/CCrawl.de-en/train.tok.de"
])
args.add_argument(
    "-vo", "--vocab-output",
    default="data/model_bpe/model.bpe_merges"
)
args.add_argument("-vs", "--vocab-size", type=int, default=8000)
args.add_argument("-n", "--number-of-lines", type=int, default=100000)
args.add_argument("-m", "--model", default="greedy")
# arguments specific to models
args.add_argument("--threshold", type=int, default=2)
args.add_argument("--seed", default=0)

args = args.parse_args()

print("Loading data")
data = []
for f in args.input:
    with open(f, "r") as f:
        data += list(f.readlines()[:args.number_of_lines])

print("Fitting model")
model = get_model(args.model)(
    seed=args.seed,
    threshold=args.threshold,
)
model.fit(data, vocab_size=args.vocab_size)
model.save(args.vocab_output)

#!/usr/bin/env python3

import argparse
from utils import get_model

args = argparse.ArgumentParser()
args.add_argument("-i", "--input", nargs="+", default=["data/CCrawl.de-en/train.tok.en", "data/CCrawl.de-en/train.tok.de"])
args.add_argument("-vo", "--vocab-output", default="data/model_bpe/model.bpe_merges")
args.add_argument("-vs", "--vocab-size", type=int, default=16392)
args.add_argument("-n", "--number-of-lines", type=int, default=100000)
args.add_argument("-m", "--model", default="greedy")
# arguments specific to models
args.add_argument("--randomness-dist", default="uniform")
args.add_argument("--randomness-temp", type=float, default=1)
args.add_argument("--greedy-n", type=int, default=4)
args.add_argument("--beam-n", type=int, default=5)
args.add_argument("--beam-n-expand", type=int, default=5)
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
    randomness_dist=args.randomness_dist,
    randomness_temp=args.randomness_temp,
    beam_n=args.beam_n,
    beam_n_expand=args.beam_n_expand,
    greedy_n=args.greedy_n,
)
model.fit(data, vocab_size=args.vocab_size)
model.save(args.vocab_output)
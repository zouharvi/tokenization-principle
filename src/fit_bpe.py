#!/usr/bin/env python3

import argparse
from bpe_models import get_bpe_model

args = argparse.ArgumentParser()
args.add_argument("-i", "--input", nargs="+", default=["data/CCrawl.de-en/train.tok.en", "data/CCrawl.de-en/train.tok.de"])
args.add_argument("-vo", "--vocab-output", default="computed/standard.bpe_model")
args.add_argument("-vs", "--vocab-size", type=int, default=4096)
args.add_argument("-n", "--number-of-lines", type=int, default=10000)
args.add_argument("-m", "--model", default="greedy")
args.add_argument("--seed", default=0)
args = args.parse_args()

print("Loading data")
data = []
for f in args.input:
    with open(f, "r") as f:
        data += list(f.readlines()[:args.number_of_lines])

print("Fitting BPE")
model = get_bpe_model(args.model)(seed=args.seed)
model.fit(data, vocab_size=args.vocab_size)
model.save(args.vocab_output)
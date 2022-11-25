#!/usr/bin/env python3

import argparse
from bpe_models.standard import StandardBPE

args = argparse.ArgumentParser()
args.add_argument("-i", "--input", nargs="+", default=["data/CCrawl.de-en/orig.en", "data/CCrawl.de-en/orig.de"])
args.add_argument("-o", "--output", default="computed/standard.bpe_model")
args.add_argument("-vs", "--vocab-size", type=int, default=4096)
args.add_argument("-n", "--number-of-lines-in-each-file", type=int, default=10000)
args = args.parse_args()

print("Loading data")
data = []
for f in args.input:
    with open(f, "r") as f:
        data += list(f.readlines()[:args.number_of_lines_in_each_file])

print("Fitting BPE")
model = StandardBPE()
model.fit(data, vocab_size=args.vocab_size)
model.save(args.output)
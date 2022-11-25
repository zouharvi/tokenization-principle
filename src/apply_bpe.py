#!/usr/bin/env python3

import argparse
from bpe_models.standard import StandardBPE

args = argparse.ArgumentParser()
args.add_argument("-i", "--input", default="data/CCrawl.de-en/orig.en")
args.add_argument("-o", "--output", default="data/CCrawl.de-en/orig.bpe.en")
args.add_argument(
    "-vi", "--vocab-input",
    default="computed/standard.bpe_model"
)
args.add_argument(
    "-n", "--number-of-lines-in-each-file",
    type=int, default=10000
)
args = args.parse_args()

print("Loading data")
with open(args.input, "r") as f:
    data = list(f.readlines()[:args.number_of_lines_in_each_file])

print("Applying BPE")
model = StandardBPE()
model.load(args.vocab_input)
data = model.encode(data)

total_subwords = sum(len(word) for line in data for word in line)
print("Outputting", total_subwords, "total subwords")

with open(args.output, "w") as f:
    for line in data:
        f.write(line + "\n")

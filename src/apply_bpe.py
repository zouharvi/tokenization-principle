#!/usr/bin/env python3

import argparse
from bpe_models.base import BaseBPE

args = argparse.ArgumentParser()
args.add_argument("-i", "--input", default="data/CCrawl.de-en/orig.en")
args.add_argument("-o", "--output",
                  default="data/CCrawl.de-en/orig.standard.en")
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
model = BaseBPE()
model.load(args.vocab_input)
data = model.encode(data)

total_subwords = sum(line.count(" ") + 1 for line in data)
print("Outputting", total_subwords, "total subwords")
total_unks = sum((" " + line).count(" UNK") for line in data)
print(
    f"Total of {total_unks} UNKs outputted",
    f"({total_unks/total_subwords:.4%} of all subwords)"
)

with open(args.output, "w") as f:
    for line in data:
        f.write(line + "\n")

# greedy     161933 (746)
# random     278583 (445)
# antigreedy 385774 (7188)

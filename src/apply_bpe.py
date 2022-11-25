#!/usr/bin/env python3

import argparse
from bpe_models.standard import StandardBPE

args = argparse.ArgumentParser()
args.add_argument("-i", "--input", default="data/CCrawl.de-en/orig.en")
args.add_argument("-o", "--output", default="data/CCrawl.de-en/orig.standard.en")
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

total_subwords = sum(word.count(" ")+1 for line in data for word in line)
print("Outputting", total_subwords, "total subwords")
total_unks = sum("UNK" in word for line in data for word in line)
print(total_unks)
print(f"UNKs represent {total_unks/total_subwords:.4%} total subwords")


with open(args.output, "w") as f:
    for line in data:
        f.write(line + "\n")

# standard 1061855
# random 1303817
# antistandard 1485704
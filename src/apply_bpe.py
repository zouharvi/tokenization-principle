#!/usr/bin/env python3

import argparse
import numpy as np
import collections
import json
from bpe_models.base import BaseBPE

args = argparse.ArgumentParser()
args.add_argument(
    "-i", "--input", default="data/CCrawl.de-en/orig.en"
)
args.add_argument(
    "-o", "--output",
    default="data/CCrawl.de-en/orig.greedy.en"
)
args.add_argument(
    "-vi", "--vocab-input",
    default="computed/greedy.bpe_model"
)
args.add_argument(
    "-n", "--number-of-lines",
    type=int, default=10000
)
args.add_argument(
    "--method", default="greedy_naive"
)
args.add_argument("--logfile", default=None)
args = args.parse_args()

def compute_entropy(data: list[str]):
    data = [word for line in data for word in line.split()]
    data = collections.Counter(data)
    total_units = sum(data.values())
    data = np.array(list(data.values()))/total_units
    entropy = -np.sum(data * np.log2(data))
    return entropy

print("Loading data")
with open(args.input, "r") as f:
    data = [x.rstrip("\n") for x in f.readlines()[:args.number_of_lines]]

print("Applying BPE")
model = BaseBPE()
model.load(args.vocab_input)
entropy_word = compute_entropy(data)
print(f"Word entropy: {entropy_word:.3f}")
data = model.encode(data, method=args.method)
entropy_subword = compute_entropy(data)

# save to file
with open(args.output, "w") as f:
    for line in data:
        f.write(line + "\n")

print(f"Subword entropy: {entropy_subword:.3f}")

total_subwords = sum(line.count(" ") + 1 for line in data)
print("Outputting", total_subwords, "total subwords")
total_unks = sum((" " + line).count(" UNK") for line in data)
print(
    f"Total of {total_unks} UNKs outputted",
    f"({total_unks/total_subwords:.4%} of all subwords)"
)

if args.logfile is not None:
    with open(args.logfile, "a") as f:
        f.write(json.dumps({
            "model": args.vocab_input.split("/")[-1],
            "total_subwords": total_subwords,
            "total_unks": total_unks,
            "number_of_lines": args.number_of_lines,
            "entropy_word": entropy_word,
            "entropy_subword": entropy_subword,
            "output": args.output,
            "input": args.input,
        })+"\n")
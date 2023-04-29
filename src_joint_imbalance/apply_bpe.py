#!/usr/bin/env python3

import argparse
import json
from bpe_models.base import BaseBPE

args = argparse.ArgumentParser()
args.add_argument(
    "-i", "--input", default="data/CCrawl.de-en/train.tok.en",
)
args.add_argument(
    "-o", "--output",
    default="data/model_bpe_greedy/train.en"
)
args.add_argument(
    "-vi", "--vocab-input",
    default="data/model_bpe_greedy/model.bpe_merges"
)
args.add_argument(
    "-n", "--number-of-lines",
    type=int, default=1000000
)
args.add_argument(
    "--method", default="greedy_naive"
)
args.add_argument("--logfile", default=None)
args = args.parse_args()

print("Loading data")
with open(args.input, "r") as f:
    data = [x.rstrip("\n") for x in f.readlines()[:args.number_of_lines]]

print("Applying BPE")
model = BaseBPE()

model.load(args.vocab_input)
data = model.encode(data, method=args.method)

# save to file
with open(args.output, "w") as f:
    for line in data:
        f.write(line + "\n")

total_subwords = sum(line.count(" ") + 1 for line in data)
print("Outputting", total_subwords, "total subwords")
total_unks = sum((" " + line).count(" UNK") for line in data)
print(
    f"Total of {total_unks} UNKs outputted",
    f"({total_unks/total_subwords:.4%} of all subwords)"
)

logline = {
    "model": args.vocab_input.split("/")[-1],
    "method": args.method,
    "vocab_size": len(model.merge_operations),
    "total_subwords": total_subwords,
    "total_unks": total_unks,
    "number_of_lines": args.number_of_lines,
    "output": args.output,
    "input": args.input,
}
print(logline)
if args.logfile is not None:
    with open(args.logfile, "a") as f:
        f.write(json.dumps(logline)+"\n")
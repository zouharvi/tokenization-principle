#!/usr/bin/env python3

import argparse
import collections
import json
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("src")
import figures.fig_utils

# rsync -azP euler:/cluster/work/sachan/vilem/random-bpe/logs/train_mt_*.log logs/

args = argparse.ArgumentParser()
args.add_argument("-i", "--input", default="computed/azaroth.jsonl")
args = args.parse_args()

with open(args.input, "r") as f:
    data_logfile = [json.loads(x) for x in f.readlines()]

SIZES = set()
total_chars = {}
total_words = {}

data = collections.defaultdict(lambda: collections.defaultdict(dict))
for line in data_logfile:
    signature = "/".join(line["output"].split("/")[1:-1]).removeprefix("model_").split("/")
    if "antigreedy" in line["output"]:
        continue
    model = signature[0]
    size = signature[1]
    SIZES.add(size)
    data[model][size][line["input"]] = line
    if "total_chars" in line:
        total_chars[line["input"]] = line["total_chars"]
        total_words[line["input"]] = line["total_words"]

data_plotting = collections.defaultdict(list)

for signature, data_local in data.items():
    for size in SIZES:
        data_size = list(data_local[size].values())
        total_subwords = sum(
            x["total_subwords"] + (x["total_unks"] if x["total_unks"] is not None else 0)
            for x in data_size
        )
        print(f"[{len(data_size)}] {signature + '(' + size + ')':>30}: {total_subwords/1000000:.1f}M")

        if len(data_size) != 6:
            continue
        data_plotting[signature].append(total_subwords)

for signature_i, (signature, values) in enumerate(data_plotting.items()):
    plt.scatter(
        list(values),
        [signature_i]*len(values),
    )

# plt.vlines(
#     x=sum(total_chars.values()),
#     ymin=0, ymax=len(data_plotting),
# )
# plt.vlines(
#     x=sum(total_words.values()),
#     ymin=0, ymax=len(data_plotting),
# )

plt.yticks(
    range(len(data_plotting)),
    data_plotting.keys(),
)
plt.ylabel("Tokenization method")
plt.xlabel("Compressed data size (lower is better)")
plt.tight_layout()
plt.show()

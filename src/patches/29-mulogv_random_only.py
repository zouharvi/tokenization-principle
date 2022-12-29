#!/usr/bin/env python3

import argparse
import collections
import json
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
sys.path.append("src")
import figures.fig_utils

# rsync -azP euler:/cluster/work/sachan/vilem/random-bpe/logs/train_mt_*.log logs/

args = argparse.ArgumentParser()
args.add_argument("-i", "--input", default="computed/glabrus.jsonl")
args = args.parse_args()

with open(args.input, "r") as f:
    data_logfile = [json.loads(x) for x in f.readlines()]

SIZES = set()

data = collections.defaultdict(lambda: collections.defaultdict(dict))
for line in data_logfile:
    signature = "/".join(line["output"].split("/")[1:-1]).removeprefix("model_").split("/")[1].split("_")
    temperature = signature[0]
    vocab_size = signature[1]
    SIZES.add(temperature)
    data[vocab_size][temperature][line["input"]] = line

data_plotting = collections.defaultdict(list)

def load_mt_bleu(temperature, vocab_size):
    filename = f"logs/train_mt_{temperature}_{vocab_size}.log"
    if not os.path.isfile(filename):
        print("skipping", filename)
        return None
    with open(filename, "r") as f:
        data = [line.split("best_bleu ")[1] for line in f.readlines() if "best_bleu" in line]
    if data:
        bleu = float(data[-1])
        return bleu
    else:
        print("skipping", filename)
        return None

for vocab_size, data_local in data.items():
    for temperature in SIZES:
        data_size = list(data_local[temperature].values())
        if len(data_size) != 6:
            continue

        total_subwords = sum(
            x["total_subwords"]
            for x in data_size
        )
        print(vocab_size, temperature)
        bleu = load_mt_bleu(temperature, vocab_size)
        if bleu is None:
            continue

        data_plotting[vocab_size].append((total_subwords, bleu))


for signature_i, (signature, values) in enumerate(data_plotting.items()):
    values.sort(key=lambda x: x[0])

    plt.plot(
        [x[0] for x in values],
        [x[1] for x in values],
        label=signature,
        marker=".",
        # [signature_i]*len(values),
    )

plt.vlines(
    x=86844987,
    ymin=35, ymax=41,
    color="black",
)
plt.vlines(
    x=19620623,
    ymin=35, ymax=41,
    color="black",
)

plt.legend()
plt.ylabel("Dev BLEU")
plt.xlabel("Compressed data size (lower is better)")
plt.tight_layout()
plt.show()

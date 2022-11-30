#!/usr/bin/env python3

import argparse
import collections
import json
import numpy as np

args = argparse.ArgumentParser()
args.add_argument("-i", "--input", default="computed/apply_bpe_beam.jsonl")
args = args.parse_args()

with open(args.input, "r") as f:
    data_logfile = [json.loads(x) for x in f.readlines()]

# model + method defines

data = collections.defaultdict(list)
for line in data_logfile:
    data[(line["model"], line["method"])].append(line)

data_flat = {}
for (model, method), data_local in data.items():
    assert len(data_local) == 2
    model = model.removesuffix(".bpe_merges")
    model = model.replace("greedy_beamsearch", "beamsearch")
    # remove vocab size
    if "_v" in model:
        model = model.split("_v")[0] 
    # use only model as a key here, may want to change later
    data_flat[model, data_local[0]["vocab_size"]] = np.average([x["total_subwords"] for x in data_local])

data_flat = list(data_flat.items())
data_flat.sort(key=lambda x: x[1])

for (model, vocab_size), total_subwords in data_flat:
    print(f"{model:>40} ({vocab_size}): {total_subwords:.0f}")
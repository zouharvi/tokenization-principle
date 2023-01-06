#!/usr/bin/env python3

import collections

import numpy as np
import fig_utils
import matplotlib.pyplot as plt
import argparse

args = argparse.ArgumentParser()
args.add_argument(
    "-db1", "--data-bpe1", nargs="+",
    default=[
        "data/model_bpe_random/005_2k/dev.de",
        "data/model_bpe_random/005_2k/dev.en",
    ])
args.add_argument(
    "-db2", "--data-bpe2", nargs="+",
    default=[
        "data/model_bpe_random/005_8k/dev.de",
        "data/model_bpe_random/005_8k/dev.en",
    ])
args.add_argument(
    "-db3", "--data-bpe3", nargs="+",
    default=[
        "data/model_bpe_random/005_32k/dev.de",
        "data/model_bpe_random/005_32k/dev.en",
    ])
args = args.parse_args()

data = []

for data_bpe_names in [args.data_bpe1, args.data_bpe2, args.data_bpe3]:
    data_bpe = list(collections.Counter([
            word
            for file in data_bpe_names
            for line in open(file, "r").readlines()
            for word in line.rstrip("\n").split(" ")
        ]).most_common())

    total_subwords = sum([freq for word, freq in data_bpe])
    data_bpe_vals = [freq/total_subwords for word, freq in data_bpe]
    data.append(data_bpe_vals)

# plt.figure()
fig, (ax1, ax2) = plt.subplots(
    1, 2, sharey=False,
    figsize=(5, 3),
)

for data_freqs, name in zip(data, ["2k", "8k", "32k"]):
    # todo fill between
    data_freqs_small = data_freqs[:100]
    ax1.fill_between(
        np.linspace(0, 1, len(data_freqs_small)),
        [0]*len(data_freqs_small),
        data_freqs_small,
        label=name,
        alpha=0.5,
    )

    data_freqs_small = data_freqs[-700:]
    ax2.fill_between(
        np.linspace(0, 1, len(data_freqs_small)),
        [0]*len(data_freqs_small),
        data_freqs_small,
        label=name,
        alpha=0.5,
    )

ax1.set_ylabel("Relative frequency")
ax1.set_xlabel("Subword distribution head")
ax2.set_xlabel("Subword distribution tail")

ax1.set_xticks([])
ax1.set_yticks([])
ax2.set_xticks([])
ax2.set_yticks([])

plt.legend()
plt.tight_layout(pad=0.2)
plt.show()

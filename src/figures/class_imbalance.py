#!/usr/bin/env python3

import collections
import fig_utils
import matplotlib.pyplot as plt
import argparse

args = argparse.ArgumentParser()
args.add_argument(
    "-dw", "--data-words", nargs="+",
    default=[
        "data/CCrawl.de-en/dev.tok.de",
        "data/CCrawl.de-en/dev.tok.en",
    ])
args.add_argument(
    "-db", "--data-bpe", nargs="+",
    default=[
        "data/model_bpe_random/005_4k/dev.de",
        "data/model_bpe_random/005_4k/dev.en",
    ])
args = args.parse_args()

# data_words = [
#     freq
#     for word, freq in collections.Counter([
#         word
#         for file in args.data_words
#         for line in open(file, "r").readlines()
#         for word in line.rstrip("\n").split(" ")
#     ]).most_common()
# ]

data_bpe = list(collections.Counter([
        word
        for file in args.data_bpe
        for line in open(file, "r").readlines()
        for word in line.rstrip("\n").split(" ")
    ]).most_common())

data_bpe_vals = [freq for word, freq in data_bpe[:120]]
print(data_bpe[:20])

plt.figure(figsize=(8, 4))
plt.bar(
    range(len(data_bpe_vals)),
    data_bpe_vals
)
plt.ylabel("Frequency")
plt.xlabel("Subword")

plt.tight_layout()
plt.show()

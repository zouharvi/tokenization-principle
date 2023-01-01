#!/usr/bin/env python3

import os
import sys
sys.path.append("src")
import figures.fig_utils
import argparse
import collections
import glob
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# rsync -azP euler:/cluster/work/sachan/vilem/random-bpe/logs/train_mt_*.log logs/

args = argparse.ArgumentParser()
args.add_argument("-d", "--data", default="data/model_bpe_random/*/dev.en")
args.add_argument("-b", "--bits", action="store_true")
args = args.parse_args()

TEMPERATURES = set()
data = collections.defaultdict(lambda: collections.defaultdict(dict))


def load_mt_bleu(temperature, vocab_size_name):
    filename = f"logs/train_mt_{temperature}_{vocab_size_name}.log"
    if not os.path.isfile(filename):
        print("skipping", filename)
        return None
    with open(filename, "r") as f:
        data = [line.split("best_bleu ")[1]
                for line in f.readlines() if "best_bleu" in line]

    if data and len(data) >= 1:
        bleu = float(data[-1])
        return bleu
    else:
        print("skipping", filename)
        return None


for fname in glob.glob(args.data):
    data_en = open(fname, "r").read()
    data_de = open(fname.replace(".en", ".de"), "r").read()
    temperature_name, vocab_size_name = fname.split("/")[-2].split("_")

    temperature = temperature_name.replace("m", "-")
    temperature = float(("^" + temperature).replace("^0",
                        "0.").replace("^-0", "-0.").removeprefix("^"))
    vocab_size = int(vocab_size_name.replace("k", "000"))

    subword_count = data_en.count(
        " ") + data_en.count("\n") + data_de.count(" ") + data_de.count("\n")
    bleu = load_mt_bleu(temperature_name, vocab_size_name)
    if not bleu:
        continue

    data[(vocab_size_name, vocab_size)][temperature] = (subword_count, bleu)

data_all = []
for signature_i, ((vocab_size_name, vocab_size), values) in enumerate(data.items()):
    values = list(values.values())
    # sort by compression
    values.sort(key=lambda x: x[0])

    if args.bits:
        xs = [x[0] * np.log2(vocab_size) for x in values]
    else:
        xs = [x[0] for x in values]

    ys = [x[1] for x in values]
    plt.plot(
        xs, ys,
        label=vocab_size_name,
        marker=".",
    )
    data_all += list(zip(xs, ys))

corr_rho, corr_pval = pearsonr([x[0] for x in data_all], [
                               x[1] for x in data_all])
print(corr_rho, corr_pval)

plt.title(f"Pearson correlation {corr_rho:.1%} (p={corr_pval:.5f})")

plt.legend()
plt.ylabel("Dev BLEU")
if args.bits:
    plt.xlabel("Bits needed for encoding (lower is better)")
else:
    plt.xlabel("Compressed data size (lower is better)")
plt.tight_layout()
plt.show()

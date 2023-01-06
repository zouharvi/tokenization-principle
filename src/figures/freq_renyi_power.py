#!/usr/bin/env python3

import argparse
import json
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
sys.path.append("src")
import figures.fig_utils

args = argparse.ArgumentParser()
args.add_argument("-i", "--input", default="computed/renyi_params.jsonl")
args = args.parse_args()

with open(args.input, "r") as f:
    data = [json.loads(x) for x in f.readlines()]

predictor_renyi = [line for line in data if line["args"]["predictor"] == "renyi"]
predictor_renyi_log = [line for line in data if line["args"]["predictor"] == "renyi_log"]
predictor_bits = [line for line in data if line["args"]["predictor"] == "bits"]
predictor_seq_len = [line for line in data if line["args"]["predictor"] == "seq_len"]
predictor_entropy = [line for line in data if line["args"]["predictor"] == "entropy"]

def plot_renyi(data, name, name_i):
    plt.plot(
        [line["args"]["power"] for line in data],
        [abs(line["pearson"]) for line in data],
        label=f"{name} Pearson",
        color=figures.fig_utils.COLORS[name_i],
    )
    plt.plot(
        [line["args"]["power"] for line in data],
        [abs(line["spearman"]) for line in data],
        label=f"{name} Spearman",
        color=figures.fig_utils.COLORS[name_i],
        linestyle="-.",
    )

plot_renyi(predictor_renyi, "Renyi", 0)
plot_renyi(predictor_renyi_log, "Renyi Log", 1)

for name_i, (name, data) in enumerate(zip(
    ["Bits", "Entropy", "Seq length"],
    [predictor_bits, predictor_entropy, predictor_seq_len]
)):
    plt.hlines(
        xmin=name_i, xmax=name_i+1,
        y=abs(data[0]["pearson"]),
        label=f"{name} Pearson",
        color=figures.fig_utils.COLORS[name_i+2],
    )
    plt.hlines(
        xmin=name_i, xmax=name_i+1,
        y=abs(data[0]["spearman"]),
        label=f"{name} Spearman",
        color=figures.fig_utils.COLORS[name_i+2],
        linestyle="-."
    )


print("ENTROPY", predictor_entropy[0])
print("RENYI", [line for line in predictor_renyi if line["args"]["power"] == 1][0])


plt.legend(
    ncol=2,

)
plt.ylabel("Correlation")
plt.xlabel("Renyi alpha")
plt.tight_layout()
plt.show()

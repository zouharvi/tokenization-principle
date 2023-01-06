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
args.add_argument("-i", "--input", default="computed/renyi_vals.jsonl")
args = args.parse_args()

with open(args.input, "r") as f:
    data = [json.loads(x) for x in f.readlines()]

predictor_renyi = [line for line in data if line["args"]["predictor"] == "renyi"]
predictor_renyi_log = [line for line in data if line["args"]["predictor"] == "renyi_log"]
# predictor_bits = [line for line in data if line["args"]["predictor"] == "bits"]
# predictor_seq_len = [line for line in data if line["args"]["predictor"] == "seq_len"]
predictor_entropy = [line for line in data if line["args"]["predictor"] == "entropy"]

def plot_renyi(data, name, name_i):
    values = [
        np.average(line["vals"])
        if name != "Renyi" or line["args"]["power"] != 1 
        else -np.average(predictor_entropy[0]["vals"])
        for line in data
    ]
    print(values)
    plt.plot(
        [line["args"]["power"] for line in data],
        values,
        label=name,
        color=figures.fig_utils.COLORS[name_i],
    )

plot_renyi(predictor_renyi, "Renyi", 0)
plot_renyi(predictor_renyi_log, "Renyi Log", 1)

# for name_i, (name, data) in enumerate(zip(
#     ["Bits", "Entropy", "Seq length"],
#     [predictor_bits, predictor_entropy, predictor_seq_len]
# )):
#     plt.hlines(
#         xmin=name_i, xmax=name_i+1,
#         y=abs(data[0]["pearson"]),
#         label=f"{name} Pearson",
#         color=figures.fig_utils.COLORS[name_i+2],
#     )
#     plt.hlines(
#         xmin=name_i, xmax=name_i+1,
#         y=abs(data[0]["spearman"]),
#         label=f"{name} Spearman",
#         color=figures.fig_utils.COLORS[name_i+2],
#         linestyle="-."
#     )

plt.legend(
    ncol=2,

)
plt.ylabel("Renyi value")
plt.xlabel("Renyi alpha")
plt.tight_layout()
plt.show()

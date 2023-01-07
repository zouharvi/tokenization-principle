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

line_entropy = [line for line in data if line["args"]["predictor"] == "entropy"][0]
line_entropy["args"]["power"] = 1
line_entropy["vals"] = [-x/512 for x in line_entropy["vals"]]
print(line_entropy["spearman"])
print(np.average(line_entropy["vals"]))
print(
    [np.average(line["vals"]) for line in data],

)
print(line_entropy)
data = [
    line if line["args"]["power"] != 1 else line_entropy
    for line in data
    if line["args"]["predictor"] == "renyi_log"
]


plt.figure(figsize=(5,3))
ax1 = plt.gca()
ax2 = ax1.twinx()

ax1.plot(
    [line["args"]["power"] for line in data],
    [(line["pearson"]) for line in data],
    label=f"Rényi Pearson",
    color=figures.fig_utils.COLORS[0],
)
ax1.plot(
    [line["args"]["power"] for line in data],
    [(line["spearman"]) for line in data],
    label=f"Rényi Spearman",
    color=figures.fig_utils.COLORS[0],
    linestyle="-.",
)
ax2.plot(
    [line["args"]["power"] for line in data],
    [np.average(line["vals"]) for line in data],
    label=f"Rényi values",
    color=figures.fig_utils.COLORS[1],
    linestyle="-.",
)

# plt.legend(
#     ncol=2,
# )

ax1.set_ylabel(r"Rényi entropy correlation with BLEU")
ax1.set_xlabel(r"$\alpha$")
ax2.set_ylabel(r"Rényi entropy value")
plt.tight_layout()
plt.savefig("computed/figures/corr_renyi_alpha.pdf")
plt.show()

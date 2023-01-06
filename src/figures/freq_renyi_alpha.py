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
args.add_argument("-i", "--input", default="computed/renyi_alphas_grid.jsonl")
args = args.parse_args()

with open(args.input, "r") as f:
    data = [json.loads(x) for x in f.readlines()]

predictor_renyi = [line for line in data if line["predictor"] == "renyi"]
predictor_bits = [line for line in data if line["predictor"] == "bits"]
predictor_seq_len = [line for line in data if line["predictor"] == "seq_len"]
predictor_freq = [line for line in data if line["predictor"] == "freq"]

plt.plot(
    [line["renyi_alpha"] for line in predictor_renyi],
    [line["pearson"] for line in predictor_renyi],
    label="Renyi Pearson",
    color=figures.fig_utils.COLORS[0],
)

plt.plot(
    [line["renyi_alpha"] for line in predictor_renyi],
    [line["spearman"] for line in predictor_renyi],
    label="Renyi Spearman",
    color=figures.fig_utils.COLORS[0],
    linestyle="-.",
)
plt.plot(
    [line["renyi_alpha"] for line in predictor_renyi],
    [abs(line["pearson"]) for line in predictor_renyi],
    alpha=0.5,
    color=figures.fig_utils.COLORS[0],
)

plt.plot(
    [line["renyi_alpha"] for line in predictor_renyi],
    [abs(line["spearman"]) for line in predictor_renyi],
    alpha=0.5,
    color=figures.fig_utils.COLORS[0],
    linestyle="-.",
)

for name_i, (name, data) in enumerate(zip(
    ["Bits", "Counts", "Seq length"],
    [predictor_bits, predictor_freq, predictor_seq_len]
)):
    plt.hlines(
        xmin=name_i, xmax=name_i+1,
        y=data[0]["pearson"], label=f"{name} Pearson",
        color=figures.fig_utils.COLORS[name_i+1],
    )
    plt.hlines(
        xmin=name_i, xmax=name_i+1,
        y=data[0]["spearman"], label=f"{name} Spearman",
        color=figures.fig_utils.COLORS[name_i+1],
        linestyle="-."
    )
    plt.hlines(
        xmin=name_i, xmax=name_i+1,
        y=abs(data[0]["pearson"]),
        color=figures.fig_utils.COLORS[name_i+1],
        alpha=0.5,
    )
    plt.hlines(
        xmin=name_i, xmax=name_i+1,
        y=abs(data[0]["spearman"]),
        color=figures.fig_utils.COLORS[name_i+1],
        alpha=0.5,
        linestyle="-."
    )

plt.legend(
    ncol=2,

)
plt.ylabel("Correlation")
plt.xlabel("Renyi alpha")
plt.tight_layout()
plt.show()

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
args.add_argument("-i", "--input", default="computed/freq_prob_renyi_alphas_grid.jsonl")
args = args.parse_args()

with open(args.input, "r") as f:
    data = [json.loads(x) for x in f.readlines()]

plt.plot(
    [line["renyi_alpha"] for line in data],
    [line["pearson"] for line in data],
    label="Renyi Pearson",
    color=figures.fig_utils.COLORS[0],
)

plt.plot(
    [line["renyi_alpha"] for line in data],
    [line["spearman"] for line in data],
    label="Renyi Spearman",
    color=figures.fig_utils.COLORS[0],
    linestyle="-.",
)
plt.hlines(
    xmin=0, xmax=1,
    y=0.539, label="Encoded bits Pearson",
    color=figures.fig_utils.COLORS[2],
)
plt.hlines(
    xmin=0, xmax=1,
    y=0.43, label="Encoded bits Spearman",
    color=figures.fig_utils.COLORS[2],
    linestyle="-."
)
plt.hlines(
    xmin=1, xmax=2,
    y=0.53, label="Gowda (counts) Pearson",
    color=figures.fig_utils.COLORS[4],
)
plt.hlines(
    xmin=1, xmax=2,
    y=0.74, label="Gowda (counts) Spearman",
    color=figures.fig_utils.COLORS[4],
    linestyle="-."
)
plt.hlines(
    xmin=2, xmax=3,
    y=0.36, label="Sequence length Pearson",
    color=figures.fig_utils.COLORS[5],
)
plt.hlines(
    xmin=2, xmax=3,
    y=0.25, label="Sequence length Spearman",
    color=figures.fig_utils.COLORS[5],
    linestyle="-."
)

plt.legend()
plt.ylabel("Correlation")
plt.xlabel("Renyi alpha")
plt.tight_layout()
plt.show()

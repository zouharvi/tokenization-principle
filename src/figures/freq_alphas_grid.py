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
args.add_argument("-i", "--input", default="computed/freq_alphas_grid.jsonl")
args = args.parse_args()

with open(args.input, "r") as f:
    data = [json.loads(x) for x in f.readlines()]

DIM = 21
image = np.full((DIM, DIM), np.nan)
plt.figure(figsize=(3.5, 1.7))

for line in data:
    s_a = int(line["start_a"] * 100 / 5)
    e_a = int(line["end_a"] * 100 / 5)
    if np.isnan(line["pearson"]):
        plt.text(
            x=e_a, y=s_a,
            s="$\cdot$",
            ha="center", va="center",
            alpha=0.5,
        )
    else:
        image[s_a][e_a] = np.abs(line["pearson"]*100)

        # minus sign
        plt.text(
            x=e_a, y=s_a+0.4,
            s="-" if line["pearson"] < 0 else "",
            ha="center", va="center",
        )

print("max spearman", max(data, key=lambda line: line["spearman"]))
print("max pearson", max(data, key=lambda line: line["pearson"]))

for e_a in range(DIM):
    for s_a in range(DIM):
        if s_a < e_a:
            plt.text(
                x=DIM-e_a-1, y=DIM-s_a-1,
                s="$\cdot$",
                ha="center", va="center",
                alpha=0.5,
            )

image = np.ma.masked_invalid(image)
cmap = matplotlib.cm.Blues.copy()
cmap.set_bad('gray', 0.35)
plt.imshow(image, cmap=cmap, aspect="auto")
BARTICKS = [30, 50, 70]
cbar = plt.colorbar(
    fraction=0.05, aspect=10,
    ticks=BARTICKS,
)
cbar.ax.set_yticklabels([f"{x}%" for x in BARTICKS])

XTICKS = [
    f"{i*5/100:.0%}" if i % 6 == 0 else ""
    for i in range(DIM)
]
YTICKS = [
    f"{i*5/100:.0%}" if i % 6 == 0 else ""
    for i in range(DIM)
]
plt.ylabel(r"Start percentile")
plt.xlabel(r"End percentile")
plt.yticks(range(DIM), YTICKS)
plt.xticks(range(DIM), XTICKS)
plt.tight_layout(pad=0.2)
plt.savefig("computed/figures/freq_alphas_grid.pdf")
plt.show()

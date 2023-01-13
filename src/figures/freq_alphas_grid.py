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

DIM = 1000
image = np.full((DIM, DIM), np.nan)
plt.figure(figsize=(3.5, 1.7))

for line in data:
    s_a = int(np.round(line["start_alpha"] * 1000))
    e_a = int(np.round(line["end_alpha"] * 1000))
    if np.isnan(line["pearson"]):
        pass
    else:
        image[s_a][e_a] = np.abs(line["pearson"]*100)

print("max spearman", max(data, key=lambda line: line["spearman"]))
print("max pearson", max(data, key=lambda line: line["pearson"]))

for e_a in range(DIM):
    for s_a in range(DIM):
        if s_a < e_a:
            pass

image = np.ma.masked_invalid(image)
cmap = matplotlib.cm.YlGn.copy()
cmap.set_bad('gray', 0.35)
plt.imshow(image, cmap=cmap, aspect="auto", interpolation="none")

# BARTICKS = [30, 50, 70]
cbar = plt.colorbar(
    fraction=0.05, aspect=10,
    # ticks=BARTICKS,
)
# cbar.ax.set_yticklabels([f"{x}%" for x in BARTICKS])

XTICKS = [
    f"{i/1000:.0%}"
    for i in range(DIM+1)
    if i % 200 == 0
]
YTICKS = [
    f"{i/1000:.0%}"
    for i in range(DIM+1)
     if i % 200 == 0
]
plt.ylabel(r"Start percentile")
plt.xlabel(r"End percentile")
plt.yticks([i*200 for i in range(len(YTICKS))], YTICKS)
plt.xticks([i*200 for i in range(len(XTICKS))], XTICKS)
plt.tight_layout(pad=0.2)
plt.savefig("computed/figures/freq_alphas_grid.pdf")
plt.show()

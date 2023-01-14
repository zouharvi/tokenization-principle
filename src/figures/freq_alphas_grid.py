#!/usr/bin/env python3

import argparse
import json
import matplotlib
import numpy as np
import matplotlib.patches as mplp
import matplotlib.pyplot as plt
import os
import sys
sys.path.append("src")
import figures.fig_utils

args = argparse.ArgumentParser()
args.add_argument("-i", "--input", default="computed/freq_prob_alphas_grid.jsonl")
args = args.parse_args()

with open(args.input, "r") as f:
    data = [json.loads(x) for x in f.readlines()]

DIM = 1000
image = np.full((DIM, DIM), np.nan)
fig = plt.figure(figsize=(3.5, 1.7))
ax1 = fig.gca()

for line in data:
    s_a = int(np.round(line["start_alpha"] * 1000))
    e_a = int(np.round(line["end_alpha"] * 1000))
    if np.isnan(line["pearson"]):
        pass
    else:
        image[s_a][e_a] = ((line["pearson"]*100))

print("max pearson", max(data, key=lambda line: abs(line["pearson"])))
print("max spearman", max(data, key=lambda line: abs(line["spearman"])))


for e_a in range(DIM):
    for s_a in range(DIM):
        if s_a < e_a:
            pass

image = np.ma.masked_invalid(image)
cmap = matplotlib.cm.YlGn.copy()
cmap.set_bad('gray', 0.05)
mappable = ax1.imshow(image, cmap=cmap, aspect="auto", interpolation="none")


# BARTICKS = [30, 50, 70]
cbar = plt.colorbar(
    mappable, ax=ax1,
    fraction=0.05, aspect=10,
    # ticks=[30, 50, 70],
)

ax2 = ax1.inset_axes([0.05, 0.07, 0.45, 0.4])
ax2.imshow(image[:30,:30], cmap=cmap, aspect="auto")
ax2.set_axis_off()
ax2.set_xticklabels([])
ax2.set_yticklabels([])

ax2.add_patch(
    mplp.Rectangle(
        (-0.5, -0.5), 30, 30,
        fill=False,
        edgecolor="black",
        linestyle="-", linewidth=1.2,
        clip_on=False,
    ),
)

ax1.add_patch(
    mplp.Rectangle(
        (-1, -1), 80, 80,
        fill=False,
        edgecolor="black",
        linestyle="-", linewidth=1.2,
        clip_on=False,
    ),
)

ax1.plot(
    [40, 300], [80, 600],
    linewidth=1.2, color="black",
    linestyle="--",
)


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
ax1.set_ylabel(r"Start percentile")
ax1.set_xlabel(r"End percentile")
ax1.set_yticks([i*200 for i in range(len(YTICKS))], YTICKS)
ax1.set_xticks([i*200 for i in range(len(XTICKS))], XTICKS)
ax1.set_xlim(0, None)
ax1.set_ylim(None, 0)
plt.tight_layout(pad=0.2)
plt.savefig("computed/figures/freq_alphas_grid.pdf")
plt.show()

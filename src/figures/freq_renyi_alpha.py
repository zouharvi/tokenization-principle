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
    label="Pearson",
)

plt.plot(
    [line["renyi_alpha"] for line in data],
    [line["spearman"] for line in data],
    label="Spearman",
)

plt.legend()
plt.ylabel("Correlation")
plt.xlabel("Renyi alpha")
plt.tight_layout()
plt.show()

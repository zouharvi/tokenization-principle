#!/usr/bin/env python3

import collections
import math
import fig_utils
import matplotlib.pyplot as plt
import argparse
import json

args = argparse.ArgumentParser()
args.add_argument("--logfile-s0", default="computed/speed_beam_s0.jsonl")
args.add_argument("--logfile-s1", default="computed/speed_beam_s1.jsonl")
args = args.parse_args()

with open(args.logfile_s0, "r") as f:
    data_s0 = [json.loads(x) for x in f.readlines()]

with open(args.logfile_s1, "r") as f:
    data_s1 = [json.loads(x) for x in f.readlines()]

plt.plot(
    [x["n"] for x in data_s0],
    [x["time"] for x in data_s0],
    marker=".", markersize=10,
    color=fig_utils.COLORS[0],
    label="Optimization 0"
)

plt.plot(
    [x["n"] for x in data_s1],
    [x["time"] for x in data_s1],
    marker=".", markersize=10,
    color=fig_utils.COLORS[1],
    label="Optimization 1"
)

plt.xlabel("$n$ beams (log)")
plt.ylabel("Time (seconds, real, log)")
plt.legend()
plt.tight_layout()
plt.show()
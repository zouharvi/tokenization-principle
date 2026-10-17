#!/usr/bin/env python3

import argparse
import collections
import glob

import tokenization_principle.joint_imbalance.figures.fig_utils as fig_utils
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st


# returns confidence interval of mean
def confidence_interval(data, conf=0.95):
  mean, sem, m = np.mean(data), st.sem(data), st.t.ppf((1+conf)/2., len(data)-1)
  return mean - m*sem, mean + m*sem

# rsync -azP euler:/cluster/work/sachan/vilem/predicting-performance/logs/train_mt_s*.log logs/

args = argparse.ArgumentParser()
args.add_argument("-l", "--logfiles", default="logs/train_mt_s*")
args = args.parse_args()

PERFORMANCES = collections.defaultdict(lambda: collections.defaultdict(list))

for fname in glob.glob(args.logfiles):
    seed, langs, balance = fname.split(
        "/")[-1].removeprefix("train_mt_").removesuffix(".log").split("_")
    balance = balance.replace("k", "000").split("-")
    balance_ratio = float(balance[0]) / (float(balance[0]) + float(balance[1]))

    with open(fname, "r") as f:
        data = [l for l in f]
    if not data:
        continue
    perf = [float(l.split("best_chrf ")[-1]) for l in data][-1]

    PERFORMANCES[langs][balance_ratio].append(perf)

# sort according to the balance ratio
PERFORMANCES = {
    k: [x for x in sorted(d.items(), key=lambda x: x[0])]
    for k, d in PERFORMANCES.items()
}

BALANCES = [
    000 / 600, 100 / 600, 300 / 600, 500 / 600, 600 / 600
]

fig = plt.figure(figsize=(4, 3))
ax = plt.gca()

for key_i, key in enumerate(["en-de", "de-en"]):
    perfs = PERFORMANCES[key]
    # average across seeds
    perfs = [
        (ratio, np.average(l), confidence_interval(l, conf=0.95))
        for ratio, l in perfs
    ]
    ratios = [x[0] for x in perfs]
    perf_cis = [x[2] for x in perfs]
    perfs = [x[1] for x in perfs]

    ax.plot(
        ratios, perfs,
        marker=fig_utils.MARKERS[0],
        label=key.upper().replace("-", r"$\rightarrow$")
    )
    ax.fill_between(
        ratios,
        [x[0] for x in perf_cis],
        [x[1] for x in perf_cis],
        color="gray",
        alpha=0.2,
    )

ax.set_ylabel("ChrF")
ax.set_xlim(-0.1, 1.1)
ax.set_xlabel("Priority", labelpad=-15)
ax.set_xticks(
    [0, 1],
    ["EN\npreference", "DE\npreference"]
)
plt.legend()
plt.tight_layout(pad=0.3)
plt.savefig("computed/figures/direction_bumps.pdf")
plt.show()

#!/usr/bin/env python3

import os
import sys
import multiprocess
sys.path.append("src")
import figures.fig_utils
import argparse
import collections
import glob
import numpy as np
import matplotlib.pyplot as plt
import scipy

# rsync -azP euler:/cluster/work/sachan/vilem/random-bpe/logs/train_mt_*.log logs/
# find -path "./logs/train_mt_*.log" -exec sh -c "grep best_bleu {} | tail -n 1" \; | wc -l
# ./src/patches/30-bleu_fig_recompute.py; ./src/patches/30-bleu_fig_recompute.py --bits

args = argparse.ArgumentParser()
args.add_argument("-d", "--data", default="data/model_bpe_random/*/dev.en")
args.add_argument("-b", "--bits", action="store_true")
args.add_argument("--ci", type=float, default=0.95)
args = args.parse_args()

TEMPERATURES = set()
data = collections.defaultdict(lambda: collections.defaultdict(dict))


def load_mt_bleu_single(temperature, vocab_size_name, suffix=""):
    global skipped_count
    filename = f"logs/train_mt{suffix}_t{temperature}_v{vocab_size_name}.log"
    if not os.path.isfile(filename):
        print("skipped", filename)
        return None
    with open(filename, "r") as f:
        data = [
            line.split("best_bleu ")[1]
            for line in f.readlines()
            if "best_bleu" in line
        ]
    if len(data) >= 5:
        bleu = float(data[-1])
        return bleu
    else:
        print("skipped", filename)
        return None


def load_mt_bleu(temperature, vocab_size_name):
    bleus = [
        load_mt_bleu_single(temperature, vocab_size_name, suffix)
        for suffix in ["_s1", "_s2"]
    ]
    bleus = [x for x in bleus if x]
    print(bleus)
    if len(bleus) < 1:
        return None
    else:
        return np.max(bleus)


def process_logfile(fname):
    data_en = open(fname, "r").read()
    data_de = open(fname.replace(".en", ".de"), "r").read()
    temperature_name, vocab_size_name = fname.split("/")[-2].split("_")

    temperature = temperature_name.replace("m", "-")
    temperature = (
        "^" + temperature
    ).replace("^0", "0.").replace("^-0", "-0.").removeprefix("^")
    temperature = float(temperature)
    vocab_size = int(vocab_size_name.replace("k", "000"))

    subword_count = (
        data_en.count(" ") + data_en.count("\n") +
        data_de.count(" ") + data_de.count("\n")
    )
    bleu = load_mt_bleu(temperature_name, vocab_size_name)
    return (vocab_size_name, vocab_size, temperature), (subword_count, bleu)


with multiprocess.Pool() as pool:
    data_flat = pool.map(process_logfile, glob.glob(args.data))

for (vocab_size_name, vocab_size, temperature), val in data_flat:
    if any([x is None for x in val]):
        continue
    data[(vocab_size_name, vocab_size)][temperature] = val

data_all = []
min_xs = np.inf
max_xs = -np.inf
data = sorted(data.items(), key=lambda x: x[0][1])
for signature_i, ((vocab_size_name, vocab_size), values) in enumerate(data):
    values = list(values.values())
    # sort by compression
    values.sort(key=lambda x: x[0])

    if args.bits:
        xs = [x[0] * np.log2(vocab_size) for x in values]
    else:
        xs = [x[0] for x in values]

    ys = [x[1] for x in values]
    plt.plot(
        xs, ys,
        label=vocab_size_name,
        marker=".",
    )
    data_all += list(zip(xs, ys))

    min_xs = min(min_xs, min(xs))
    max_xs = max(max_xs, max(xs))

data_all_y = [x[1] for x in data_all]
data_all_x = [x[0] for x in data_all]
corr_rho, corr_pval = scipy.stats.pearsonr(data_all_x, data_all_y)


def linear_regression_ci(x, y, ci=0.95):
    """
    Adapted from https://gist.github.com/riccardoscalco/5356167
    """
    alpha = 1 - ci
    x = np.array(x)
    y = np.array(y)
    n = len(x)
    xy = x * y
    xx = x * x

    # y = a*x + b
    coef_a = (xy.mean() - x.mean() * y.mean()) / (xx.mean() - x.mean()**2)
    coef_b = y.mean() - coef_a * x.mean()
    s2 = 1 / n * sum([
        (y[i] - coef_b - coef_a * x[i]) ** 2 for i in np.arange(n)
    ])

    # multi-side is alpha/2
    # we want to compare only one side, so only alpha
    c = -1 * scipy.stats.t.ppf(alpha, n - 2)
    coef_a_err = c * (s2 / ((n - 2) * (xx.mean() - (x.mean())**2)))**0.5

    coef_b_err = c * (
        (s2 / (n - 2)) * (1 + (x.mean()) ** 2 / (xx.mean() - (x.mean())**2))
    )**0.5
    return (coef_a, coef_b), (coef_a - coef_a_err, coef_b - coef_b_err), (coef_a + coef_a_err, coef_b + coef_b_err)


coefs, coefs_low, coefs_high = linear_regression_ci(
    data_all_x, data_all_y, ci=args.ci)

# linear_model = np.polyfit(
#     data_all_x, data_all_y, 1
# )
# linear_model_fn = np.poly1d(linear_model)
# linear_model_fn(lin_model_xs),
lin_model_xs = np.linspace(min_xs, max_xs, 10)
plt.plot(
    lin_model_xs,
    coefs[0] * lin_model_xs + coefs[1],
    color="black",
    linestyle=":",
    zorder=-10,
)

plt.fill_between(
    lin_model_xs,
    coefs_low[0] * lin_model_xs + coefs_low[1],
    coefs_high[0] * lin_model_xs + coefs_high[1],
    color="black",
    linestyle=":",
    alpha=0.1,
    zorder=-10,
)

plt.title(
    f"Pearson correlation {corr_rho:.1%} (p={corr_pval:.6f}) | {args.ci:.0%} CI band")
plt.legend(ncol=2)
plt.ylabel("Dev BLEU")
if args.bits:
    plt.xlabel("Bits needed for encoding (lower is better)")
else:
    plt.xlabel("Compressed data size (lower is better)")
plt.tight_layout()
plt.show()

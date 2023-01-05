#!/usr/bin/env python3

import json
import os
import pickle
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
# ./src/patches/30-bleu_fig_recompute.py --predictor seq_len
# ./src/patches/30-bleu_fig_recompute.py --predictor subwords
# ./src/patches/30-bleu_fig_recompute.py --predictor bits
# ./src/patches/30-bleu_fig_recompute.py --predictor freq --freq-alpha-start 0.80 --freq-alpha-end 0.90 --use-cache;

args = argparse.ArgumentParser()
args.add_argument("-d", "--data", default="data/model_bpe_random/*/dev.en")
args.add_argument("-p", "--predictor", default="mu")
args.add_argument("--ci", type=float, default=0.95)
args.add_argument("--freq-alpha-start", type=float, default=0.65)
args.add_argument("--freq-alpha-end", type=float, default=1.00)
args.add_argument("--use-cache", action="store_true")
args = args.parse_args()

TEMPERATURES = set()
data = collections.defaultdict(lambda: collections.defaultdict(dict))


def get_prediction(data, vocab_size):
    if args.predictor in {"subwords", "mu"}:
        return data.count(" ") + data.count("\n")
    elif args.predictor in {"seq_len"}:
        return np.average([line.count(" ") + 1 for line in data.split("\n")])
    elif args.predictor in {"mu log v", "bits"}:
        return (data.count(" ") + data.count("\n")) * np.log2(vocab_size)
    elif args.predictor in {"freq"}:
        words_freqs = list(collections.Counter(data.split()).most_common())
        percentiles = np.arange(
            args.freq_alpha_start,
            # add epsilon to be included
            args.freq_alpha_end + 0.001, step=0.05
        )
        freqs = np.average([
            words_freqs[
                min(int(len(words_freqs) * percentile), len(words_freqs) - 1)
            ][1]
            for percentile in percentiles
        ])
        # freq_95 = words_freqs[index_85][1]+words_freqs[index_80][1]
        # print(words_freqs[index_95], words_freqs[:10])
        return freqs
    elif args.predictor in {"freq_prob"}:
        words_freqs = list(collections.Counter(data.split()).most_common())
        total_subwords = sum([x[1] for x in words_freqs])
        percentiles = np.arange(
            args.freq_alpha_start,
            # add epsilon to be included
            args.freq_alpha_end + 0.001, step=0.05
        )
        freqs = np.sum([
            words_freqs[
                min(int(len(words_freqs) * percentile), len(words_freqs) - 1)
            ][1]
            for percentile in percentiles
        ]) / total_subwords
        return freqs / np.log2(vocab_size)
    else:
        raise Exception("Unknown predictor " + args.predictor)


def get_predictor_title():
    if args.predictor in {"subwords", "mu"}:
        return "Compressed data size"
    elif args.predictor in {"seq_len"}:
        return "Average sequence length"
    elif args.predictor in {"mu log v", "bits"}:
        return "Bits needed for encoding"
    else:
        return "Predictor doesn't have axis label assigned yet"


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
    if data:
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

    bleu = load_mt_bleu(temperature_name, vocab_size_name)
    return (vocab_size_name, temperature), ((data_en + data_de, vocab_size), bleu)


if args.use_cache:
    data_flat = pickle.load(open("computed/bleu_corr_cache.pkl", "rb"))
else:
    with multiprocess.Pool() as pool:
        data_flat = pool.map(process_logfile, glob.glob(args.data))
    pickle.dump(data_flat, open("computed/bleu_corr_cache.pkl", "wb"))

with multiprocess.Pool() as pool:
    data_flat = pool.map(
        lambda x: (x[0], (get_prediction(x[1][0][0], x[1][0][1]), x[1][1])),
        data_flat
    )


for (vocab_size_name, temperature), val in data_flat:
    if any([x is None for x in val]):
        continue
    data[vocab_size_name][temperature] = val

data_all = []
min_xs = np.inf
max_xs = -np.inf

plt.figure(figsize=(4, 3.5))

data = sorted(data.items(), key=lambda x: int(x[0].replace("k", "000")))

for signature_i, (vocab_size_name, values) in enumerate(data):
    values = list(values.values())
    # sort by compression
    values.sort(key=lambda x: x[0])

    # remove outliers
    values = [x for x in values if x[1] >= 33]
    while True:
        values_new = [values[0]] + [
            values[i] for i in range(1, len(values))
            if values[i][1] >= values[i - 1][1] - 1.8
        ]
        if values_new == values:
            break
        else:
            values = values_new

    xs = [x[0] for x in values]
    ys = [x[1] for x in values]

    plt.plot(
        xs, ys,
        label=r"V=" + vocab_size_name,
        marker=".",
    )
    data_all += list(zip(xs, ys))

    min_xs = min(min_xs, min(xs))
    max_xs = max(max_xs, max(xs))

data_all_y = [x[1] for x in data_all]
data_all_x = [x[0] for x in data_all]
corr_pearson_rho, corr_pearson_pval = scipy.stats.pearsonr(
    data_all_x, data_all_y)
corr_spearman_rho, corr_spearman_pval = scipy.stats.spearmanr(
    data_all_x, data_all_y)


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

if args.predictor == "freq":
    ADDITIONAL_SIGNATURE = {
        "start_a": args.freq_alpha_start, "end_a": args.freq_alpha_end}
else:
    ADDITIONAL_SIGNATURE = {}

print(
    "JSON!",
    json.dumps({
        "pearson": corr_pearson_rho, "spearman": corr_spearman_rho,
        "pearson_p": corr_pearson_pval, "spearman_p": corr_spearman_pval,
    } | ADDITIONAL_SIGNATURE),
    sep="",
)
plt.title(
    f"Pearson correlation {corr_pearson_rho:.1%} (p={corr_pearson_pval:.4f})\n" +
    f"Spearman correlation {corr_spearman_rho:.1%} (p={corr_spearman_pval:.4f})"
)
plt.legend(
    ncol=5,
    loc="upper center",
    labelspacing=0.0,
    handlelength=0.9,
    handletextpad=0.25,
    columnspacing=0.6,
)
# add space for legend
plt.ylim(min(data_all_y) - 0.2, max(data_all_y) + 0.85)
plt.ylabel("Dev BLEU")
plt.xlabel(get_predictor_title())
plt.tight_layout()

plt.savefig(
    "computed/figures/bleu_corr_" +
    args.predictor.replace(" ", "_") + ".pdf"
)
plt.show()

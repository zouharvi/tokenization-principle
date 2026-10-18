#!/usr/bin/env python3

import argparse
import collections
import glob
import json
import os
import pickle
import sys

import matplotlib.pyplot as plt
import multiprocess
import numpy as np
import scipy

from tokenization_scorer import score, METRIC_TITLES

# rsync -azP euler:/cluster/work/sachan/vilem/random-bpe/logs/train_mt_*.log logs/
# ./tokenization_principle/figures/performance_predict_multi.py --predictor bits --write-cache
# ./tokenization_principle/figures/performance_predict_multi.py --predictor seq_len --write-cache
# ./tokenization_principle/figures/performance_predict_multi.py --predictor renyi_eff --power 3.0 --load-cache


args = argparse.ArgumentParser()
args.add_argument("-d", "--data", default="data/*/*/dev.en")
args.add_argument("-p", "--predictor", default="bits")
args.add_argument("--load-cache", action="store_true")
args.add_argument("--no-graphics", action="store_true")
args.add_argument("--write-cache", action="store_true")
args.add_argument("--ci", type=float, default=0.95)
args.add_argument("--freq-alpha-start", type=float, default=0.0)
args.add_argument("--freq-alpha-end", type=float, default=1.0)
args.add_argument("--power", type=float, default=None)
args.add_argument("--central-measure", default=None)
args.add_argument("--base-vals", default=None)
args.add_argument("--aggregator", default=None)
args.add_argument("--data-out", default=None)
args = args.parse_args()
args_kwargs = vars(args)

predictor_title = METRIC_TITLES.get(args.predictor, args.predictor.replace("_", " ").capitalize())

TEMPERATURES = set()
data = collections.defaultdict(lambda: collections.defaultdict(dict))


def load_mt_perf_single(model_name, train_lines_name, vocab_size_name, suffix=""):
    filename = f"logs/train_mt_{model_name}{suffix}_l{train_lines_name}_v{vocab_size_name}.log"
    if not os.path.isfile(filename):
        # print("skipped", filename)
        return None
    with open(filename, "r") as f:
        data = [
            line.split("best_chrf ")[1]
            for line in f
            if "best_chrf" in line
        ]
    if data:
        perf = float(data[-1])
        return perf
    else:
        # print("skipped", filename)
        return None


def load_mt_perf(model_name, train_lines_name, vocab_size_name):
    perfs = [
        load_mt_perf_single(
            model_name, train_lines_name,
            vocab_size_name, suffix
        )
        for suffix in ["_s1", "_s2", "_s3", "_s4", "_s5"]
    ]
    perfs = [x for x in perfs if x]
    if len(perfs) < 1:
        return None
    else:
        print(perfs)
        return np.max(perfs)


def process_logfile(fname):
    if "bpe_random" in fname:
        return None
    with open(fname, "r") as f:
        data_en = f.read()
    with open(fname.replace(".en", ".de"), "r") as f:
        data_de = f.read()
    train_lines_name, vocab_size_name = fname.split("/")[-2].split("_")
    train_lines_name = train_lines_name.removeprefix("l")
    model_name = fname.split("/")[-3].removeprefix("model_")

    vocab_size = int(vocab_size_name.replace("k", "000"))

    perf = load_mt_perf(model_name, train_lines_name, vocab_size_name)
    if not perf:
        return None
    model_name = model_name.removeprefix("tokenizer_")
    return (model_name, vocab_size_name, train_lines_name), ((data_en + data_de, vocab_size), perf)


if args.load_cache:
    with open("computed/perf_corr_cache_multi.pkl", "rb") as f:
        data_flat = pickle.load(f)
else:
    with multiprocess.Pool() as pool:
        data_flat = pool.map(process_logfile, glob.glob(args.data))
        data_flat = [x for x in data_flat if x is not None]
    if args.write_cache:
        with open("computed/perf_corr_cache_multi.pkl", "wb") as f:
            pickle.dump(data_flat, f)


def compute_metric(item):
    (model_name, vocab_size_name, train_lines_name), ((text, vocab_size), perf) = item
    score_kwargs = {
        "vocab": vocab_size,
        "verbose": False,
    }
    if args.power is not None:
        score_kwargs["power"] = args.power
    if args.freq_alpha_start is not None:
        score_kwargs["perc_start"] = args.freq_alpha_start
    if args.freq_alpha_end is not None:
        score_kwargs["perc_end"] = args.freq_alpha_end
    if args.central_measure is not None:
        score_kwargs["central_measure"] = args.central_measure
    if args.aggregator is not None:
        score_kwargs["aggregator"] = args.aggregator
    if args.base_vals is not None:
        score_kwargs["base_vals"] = args.base_vals

    metric_val = score(text, metric=args.predictor, **score_kwargs)
    return (model_name, vocab_size_name, train_lines_name), (metric_val, perf)

with multiprocess.Pool() as pool:
    data_flat = pool.map(compute_metric, data_flat)


for (model_name, vocab_size_name, train_lines_name), val in data_flat:
    data[model_name][(vocab_size_name, train_lines_name)] = val

data_all = []
min_xs = np.inf
max_xs = -np.inf

plt.figure(figsize=(6, 5.5))

# sort by model name
data = sorted(data.items(), key=lambda x: x[0])

for signature_i, (model_name, values) in enumerate(data):
    values = list(values.values())
    # sort by compression
    values.sort(key=lambda x: x[0])

    # remove outliers
    values = [x for x in values if x[1] >= 30]
    while True:
        values_new = [values[0]] + [
            values[i] for i in range(1, len(values))
            if values[i][1] >= values[i - 1][1] - 4
        ]
        if values_new == values:
            break
        else:
            values = values_new


    xs = [x[0] for x in values]
    ys = [x[1] for x in values]

    plt.plot(
        xs, ys,
        label=model_name,
        marker=".",
    )

    # for further processing
    data_all += list(zip(xs, ys))

    min_xs = min(min_xs, min(xs))
    max_xs = max(max_xs, max(xs))

data_all_y = [x[1] for x in data_all]
data_all_x = [x[0] for x in data_all]
corr_pearson_rho, corr_pearson_pval = scipy.stats.pearsonr(
    data_all_x, data_all_y
)
corr_spearman_rho, corr_spearman_pval = scipy.stats.spearmanr(
    data_all_x, data_all_y
)


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
    data_all_x, data_all_y,
    ci=args.ci
)

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

# print section
data_all_x, data_all_y = zip(
    *sorted(zip(data_all_x, data_all_y), key=lambda x: x[0]))
print(
    "JSON!",
    json.dumps({
        "pearson": corr_pearson_rho, "spearman": corr_spearman_rho,
        "pearson_p": corr_pearson_pval, "spearman_p": corr_spearman_pval,
        "args": args_kwargs,
        "perfs": data_all_y,
        "vals": data_all_x
    }),
    sep="",
)

if args.no_graphics:
    sys.exit()


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
plt.ylabel("Dev chrF")
plt.xlabel(predictor_title)
plt.tight_layout(pad=0)

plt.savefig(
    "computed/figures/perf_corr_" +
    args.predictor.replace(" ", "_") + ".pdf"
)
plt.show()

#!/usr/bin/env python3

import argparse
import collections
import glob
import json
import os
import pickle
import sys

import matplotlib.pyplot as plt
from multiprocess.pool import Pool
import numpy as np
import scipy
from sklearn.model_selection import train_test_split
from tokenization_scorer import METRIC_TITLES, score

# rsync -azP euler:/cluster/work/sachan/vilem/random-bpe/logs/train_mt_*.log logs/
# ./tokenization_principle/figures/performance_predict.py --predictor entropy --load-cache
# ./tokenization_principle/figures/performance_predict.py --predictor entropy_eff --load-cache
# ./tokenization_principle/figures/performance_predict.py --predictor seq_len --load-cache
# ./tokenization_principle/figures/performance_predict.py --predictor renyi --power 3.0 --load-cache
# ./tokenization_principle/figures/performance_predict.py --predictor renyi_eff --power 2.5 --load-cache
# ./tokenization_principle/figures/performance_predict.py --predictor gowda --freq-alpha-start 0.03 --freq-alpha-end 0.83 --load-cache


parser = argparse.ArgumentParser()
parser.add_argument("-d", "--data", default="data/model_bpe_random/*/dev.en")
parser.add_argument("-p", "--predictor", default="bits")
parser.add_argument("--load-cache", action="store_true")
parser.add_argument("--no-graphics", action="store_true")
parser.add_argument("--data-train", action="store_true")
parser.add_argument("--write-cache", action="store_true")
parser.add_argument("--ci", type=float, default=0.95)
parser.add_argument("--freq-alpha-start", type=float, default=0.0)
parser.add_argument("--freq-alpha-end", type=float, default=1.0)
parser.add_argument("--power", type=float, default=None)
parser.add_argument("--central-measure", default=None)
parser.add_argument("--base-vals", default=None)
parser.add_argument("--aggregator", default=None)
args = parser.parse_args()
args_kwargs = vars(args)

predictor_title: str = str(METRIC_TITLES.get(args.predictor) or args.predictor.replace("_", " ").capitalize())

TEMPERATURES = set()
data = collections.defaultdict(lambda: collections.defaultdict(dict))


def load_mt_perf_single(temperature, vocab_size_name, suffix=""):
    # TODO: for `repl`, switched to sacrebleu
    filename = f"logs/train_mt{suffix}_t{temperature}_v{vocab_size_name}.log"
    if not os.path.isfile(filename):
        print("skipped", filename)
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
        print("skipped", filename)
        return None


def load_mt_perf(temperature, vocab_size_name):
    perfs = [
        load_mt_perf_single(temperature, vocab_size_name, suffix)
        for suffix in ["_s1", "_s2", "_s3", "_s4", "_s5"]
    ]
    perfs = [x for x in perfs if x]
    print(perfs)
    if len(perfs) < 1:
        return None
    else:
        return np.max(perfs)


def process_logfile(fname):
    with open(fname, "r") as f:
        data_en = f.read()
    with open(fname.replace(".en", ".de"), "r") as f:
        data_de = f.read()
    temperature_name, vocab_size_name = fname.split("/")[-2].split("_")

    temperature = temperature_name.replace("m", "-")
    temperature = (
        "^" + temperature
    ).replace("^0", "0.").replace("^-0", "-0.").removeprefix("^")
    temperature = float(temperature)
    vocab_size = int(vocab_size_name.replace("k", "000"))

    perf = load_mt_perf(temperature_name, vocab_size_name)
    return (vocab_size_name, temperature), ((data_en + data_de, vocab_size), perf)


if args.load_cache:
    with open("computed/perf_corr_cache.pkl", "rb") as f:
        data_flat = pickle.load(f)
else:
    with Pool() as pool:
        data_flat = pool.map(process_logfile, glob.glob(args.data))
    if args.write_cache:
        with open("computed/perf_corr_cache.pkl", "wb") as f:
            pickle.dump(data_flat, f)

def compute_metric(item):
    (vocab_size_name, temp), ((text, vocab_size), perf) = item
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
    return (vocab_size_name, temp), (metric_val, perf)

with Pool() as pool:
    data_flat = pool.map(compute_metric, data_flat)


for (vocab_size_name, temperature), val in data_flat:
    if any(x is None for x in val):
        continue
    data[vocab_size_name][temperature] = val

data_all = []
min_xs = np.inf
max_xs = -np.inf

plt.figure(figsize=(4.1, 3))

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

data_train_x, data_dev_x, data_train_y, data_dev_y = train_test_split(
    data_all_x, data_all_y, test_size=0.5,
    random_state=0
)
if args.data_train:
    data_dev_x = data_train_x
    data_dev_y = data_train_y
corr_pearson_rho, corr_pearson_pval = scipy.stats.pearsonr(
    data_dev_x, data_dev_y
)
corr_spearman_rho, corr_spearman_pval = scipy.stats.spearmanr(
    data_dev_x, data_dev_y
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
    tstat = scipy.stats.t.ppf(alpha, n - 2)
    c = -1 * tstat
    coef_a_err = c * (s2 / ((n - 2) * (xx.mean() - (x.mean())**2)))**0.5

    coef_b_err = c * (
        (s2 / (n - 2)) * (1 + (x.mean()) ** 2 / (xx.mean() - (x.mean())**2))
    )**0.5
    return (coef_a, coef_b), (coef_a - coef_a_err, coef_b - coef_b_err), (coef_a + coef_a_err, coef_b + coef_b_err), tstat



coefs, coefs_low, coefs_high, tstat = linear_regression_ci(
    data_train_x, data_train_y,
    ci=args.ci
)


data_pred_y = coefs[0] * np.array(data_dev_x) + coefs[1],
data_dev_y = np.array(data_dev_y)
mse = ((data_dev_y - data_pred_y)**2).mean()
mae = (np.abs(data_dev_y - data_pred_y)).mean()
print(
    "BAND!",
    json.dumps({
        "mse": mse,
        "mae": mae,
        "tstat": tstat,
        "a_lower": float(coefs_low[0]),
        "a_higher": float(coefs_high[0]),
        "b_lower": float(coefs_low[1]),
        "b_higher": float(coefs_high[1]),
    }),
    sep="",
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

if args.no_graphics:
    TEXT_LATEX=\
        f"{corr_pearson_rho:.2f}" + r" {\small(=" + f"{corr_pearson_pval:.3f}" + r")} & " \
        f"{corr_spearman_rho:.2f}" + r" {\small(=" + f"{corr_spearman_pval:.3f}" + r")}"
    print("LATEX!", TEXT_LATEX.replace("%", r"\%").replace("=0.000", "<0.001"))
    
# print section
data_all_x, data_all_y = zip(
    *sorted(zip(data_all_x, data_all_y), key=lambda x: x[0])
)
print(
    "JSON!",
    json.dumps({
        "pearson": float(corr_pearson_rho), "spearman": float(corr_spearman_rho),
        "pearson_p": float(corr_pearson_pval), "spearman_p": float(corr_spearman_pval),
        "args": args_kwargs,
        "perfs": data_all_y,
        "vals": [float(x) for x in data_all_x]
    }),
    sep="",
)

if args.no_graphics:
    sys.exit()

if corr_pearson_pval < 0.001:
    corr_pearson_pval = "$<$0.001"
else:
    corr_pearson_pval = f"={corr_pearson_pval:.3f}"
if corr_spearman_pval < 0.001:
    corr_spearman_pval = "$<$0.001"
else:
    corr_spearman_pval = f"={corr_spearman_pval:.3f}"


plt.title(
    f"Pearson correlation {corr_pearson_rho:.2f} (p{corr_pearson_pval})\n" \
    f"Spearman correlation {corr_spearman_rho:.2f} (p{corr_spearman_pval})"
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
plt.tight_layout(pad=0.1)

plt.savefig(
    "computed/figures/perf_corr_" +
    args.predictor.replace(" ", "_") + ".pdf"
)
plt.show()

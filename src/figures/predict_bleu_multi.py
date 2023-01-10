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
from predictors_bleu import get_predictor

# rsync -azP euler:/cluster/work/sachan/vilem/random-bpe/logs/train_mt_*.log logs/
# find -path "./logs/train_mt_*.log" -exec sh -c "grep best_bleu {} | tail -n 1" \; | wc -l
# ./src/figures/predict_bleu.py --predictor bits --write-cache
# ./src/figures/predict_bleu.py --predictor seq_len
# ./src/figures/predict_bleu.py --predictor freq --freq-alpha-start 0.80 --freq-alpha-end 0.90 --load-cache
# ./src/figures/predict_bleu.py --predictor freq_prob --freq-alpha-start 0.25 --freq-alpha-end 0.75 --load-cache
# ./src/figures/predict_bleu.py --predictor freq_prob_square --freq-alpha-start 0.25 --freq-alpha-end 0.75 --load-cache


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
args = args.parse_args()
args_kwargs = vars(args)

predictor, predictor_title = get_predictor(args.predictor)

TEMPERATURES = set()
data = collections.defaultdict(lambda: collections.defaultdict(dict))


def load_mt_bleu_single(model_name, train_lines_name, vocab_size_name, suffix=""):
    filename = f"logs/train_mt_{model_name}{suffix}_l{train_lines_name}_v{vocab_size_name}.log"
    if not os.path.isfile(filename):
        # print("skipped", filename)
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
        # print("skipped", filename)
        return None


def load_mt_bleu(model_name, train_lines_name, vocab_size_name):
    bleus = [
        load_mt_bleu_single(
            model_name, train_lines_name,
            vocab_size_name, suffix
        )
        for suffix in ["_s1", "_s2"]
    ]
    bleus = [x for x in bleus if x]
    if len(bleus) < 1:
        return None
    else:
        print(bleus)
        return np.max(bleus)


def process_logfile(fname):
    if "bpe_random" in fname:
        return None
    data_en = open(fname, "r").read()
    data_de = open(fname.replace(".en", ".de"), "r").read()
    train_lines_name, vocab_size_name = fname.split("/")[-2].split("_")
    train_lines_name = train_lines_name.removeprefix("l")
    model_name = fname.split("/")[-3].removeprefix("model_")

    vocab_size = int(vocab_size_name.replace("k", "000"))

    bleu = load_mt_bleu(model_name, train_lines_name, vocab_size_name)
    if not bleu:
        return None
    model_name = model_name.removeprefix("tokenizer_")
    return (model_name, vocab_size_name, train_lines_name), ((data_en + data_de, vocab_size), bleu)


if args.load_cache:
    data_flat = pickle.load(open("computed/bleu_corr_cache_multi.pkl", "rb"))
else:
    with multiprocess.Pool() as pool:
        data_flat = pool.map(process_logfile, glob.glob(args.data))
        data_flat = [x for x in data_flat if x is not None]
    if args.write_cache:
        pickle.dump(data_flat, open("computed/bleu_corr_cache_multi.pkl", "wb"))


with multiprocess.Pool() as pool:
    data_flat = pool.map(
        lambda x: (
            # signature stuff
            x[0], (
                # data, vocab_size, args
                predictor(x[1][0][0], x[1][0][1], args_kwargs),
                # bleu
                x[1][1]
            )
        ),
        data_flat
    )


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

    xs = [x[0] for x in values]
    ys = [x[1] for x in values]

    if model_name == "morfessor":
        print("LEN", len(values))

    plt.plot(
        xs, ys,
        label=model_name,
        marker=".",
    )
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
        "bleus": data_all_y,
        "vals": data_all_x
    }),
    sep="",
)

if args.no_graphics:
    exit()

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
plt.xlabel(predictor_title)
plt.tight_layout(pad=0)

plt.savefig(
    "computed/figures/bleu_corr_" +
    args.predictor.replace(" ", "_") + ".pdf"
)
plt.show()

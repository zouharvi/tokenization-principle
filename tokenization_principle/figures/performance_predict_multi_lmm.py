#!/usr/bin/env python3

import argparse
import collections
import glob
import os
import pickle

import multiprocess
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import pearsonr, spearmanr

from tokenization_scorer import score, METRIC_TITLES

# rsync -azP euler:/cluster/work/sachan/vilem/random-bpe/logs/train_mt_*.log logs/
# ./tokenization_principle/figures/performance_predict_multi_lmm.py --predictor seq_len --write-cache
# ./tokenization_principle/figures/performance_predict_multi_lmm.py --predictor bits --load-cache
# ./tokenization_principle/figures/performance_predict_multi_lmm.py --predictor renyi --power 3.0 --load-cache
# ./tokenization_principle/figures/performance_predict_multi_lmm.py --predictor renyi_eff --power 3.0 --load-cache
# ./tokenization_principle/figures/performance_predict_multi_lmm.py --predictor gowda --freq-alpha-start 0.03 --freq-alpha-end 0.83 --load-cache


args = argparse.ArgumentParser()
args.add_argument("-d", "--data", default="data/*/*/dev.en")
args.add_argument("-p", "--predictor", default="bits")
args.add_argument("--load-cache", action="store_true")
args.add_argument("--write-cache", action="store_true")
args.add_argument("--write-cache-name", default="")
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

data = collections.defaultdict(lambda: collections.defaultdict(dict))

def load_mt_perf_single(model_name, train_lines_name, vocab_size_name, suffix=""):
    filename = f"logs/train_mt_{model_name}{suffix}_l{train_lines_name}_v{vocab_size_name}.log"
    if not os.path.isfile(filename):
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
        return None


def load_mt_perf(model_name, train_lines_name, vocab_size_name):
    perfs = [
        load_mt_perf_single(
            model_name, train_lines_name,
            vocab_size_name, suffix
        )
        for suffix in ["_s1", "_s2"]
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
data_out = []
min_xs = np.inf
max_xs = -np.inf

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
            if values[i][1] >= values[i - 1][1] - 2
        ]
        if values_new == values:
            break
        else:
            values = values_new


    xs = [x[0] for x in values]
    ys = [x[1] for x in values]

    # for further processing
    data_all += list(zip(xs, ys))
    data_out += [(model_name, x, y) for x, y in zip(xs, ys)]

    min_xs = min(min_xs, min(xs))
    max_xs = max(max_xs, max(xs))

df = pd.DataFrame(data_out, columns =["model", "pred", "perf"])
md = smf.mixedlm("perf ~ pred + model", df, groups=df["model"])
mdf = md.fit()
print(mdf.summary())
print("pred (pval)", mdf.pvalues["pred"])
# print(mdf.conf_int())
print(mdf.normalized_cov_params)

# https://www.statsmodels.org/stable/generated/statsmodels.regression.mixed_linear_model.MixedLMResults.html
y_pred = mdf.predict()
print("Pearson  (corr, p)", *pearsonr(y_pred, df["perf"]))
print("Spearman (corr, p)", *spearmanr(y_pred, df["perf"]))


corr_pearson_mixed_rho, corr_pearson_mixed_pval = spearmanr(y_pred, df["perf"])
corr_pearson_all_rho, corr_pearson_all_pval = spearmanr(df["pred"], df["perf"])
TEXT_LATEX=\
    f"{corr_pearson_mixed_rho:.1%}" + r" {\small(=" + f"{corr_pearson_mixed_pval:.3f}" + r")} & " \
    f"{corr_pearson_all_rho:.1%}" + r" {\small(=" + f"{corr_pearson_all_pval:.3f}" + r")}"
print("LATEX!", TEXT_LATEX.replace("%", r"\%").replace("=0.000", "<0.001"))
#!/usr/bin/env python3

import json
import os
import pickle
import sys
import multiprocess
sys.path.append("src")
import argparse
import collections
import glob
import numpy as np
import statsmodels
import statsmodels.formula.api as smf
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from predictors_bleu import get_predictor

# rsync -azP euler:/cluster/work/sachan/vilem/random-bpe/logs/train_mt_*.log logs/
# ./src/figures/predict_bleu_multi_lmm.py --predictor seq_len --write-cache
# ./src/figures/predict_bleu_multi_lmm.py --predictor bits --load-cache
# ./src/figures/predict_bleu_multi_lmm.py --predictor renyi --power 3.0 --load-cache
# ./src/figures/predict_bleu_multi_lmm.py --predictor renyi_eff --power 3.0 --load-cache
# ./src/figures/predict_bleu_multi_lmm.py --predictor freq --freq-alpha-start 0.90 --freq-alpha-end 0.942 --power 1 --load-cache


args = argparse.ArgumentParser()
args.add_argument("-d", "--data", default="data/*/*/dev.en")
args.add_argument("-p", "--predictor", default="bits")
args.add_argument("--load-cache", action="store_true")
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

predictor, predictor_title = get_predictor(args.predictor)

data = collections.defaultdict(lambda: collections.defaultdict(dict))

def load_mt_bleu_single(model_name, train_lines_name, vocab_size_name, suffix=""):
    filename = f"logs/train_mt_{model_name}{suffix}_l{train_lines_name}_v{vocab_size_name}.log"
    if not os.path.isfile(filename):
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

df = pd.DataFrame(data_out, columns =["model", "pred", "bleu"])
md = smf.mixedlm("bleu ~ pred + model", df, groups=df["model"])
mdf = md.fit()
print(mdf.summary())
print("pred (pval)", mdf.pvalues["pred"])
# print(mdf.conf_int())
print(mdf.normalized_cov_params)

# https://www.statsmodels.org/stable/generated/statsmodels.regression.mixed_linear_model.MixedLMResults.html
y_pred = mdf.predict()
print("Pearson  (corr, p)", *pearsonr(y_pred, df["bleu"]))
print("Spearman (corr, p)", *spearmanr(y_pred, df["bleu"]))


corr_pearson_mixed_rho, corr_pearson_mixed_pval = spearmanr(y_pred, df["bleu"])
corr_pearson_all_rho, corr_pearson_all_pval = spearmanr(df["pred"], df["bleu"])
TEXT_LATEX=\
    f"{corr_pearson_mixed_rho:.1%}" + r" {\small(=" + f"{corr_pearson_mixed_pval:.3f}" + r")} & " \
    f"{corr_pearson_all_rho:.1%}" + r" {\small(=" + f"{corr_pearson_all_pval:.3f}" + r")}"
print("LATEX!", TEXT_LATEX.replace("%", r"\%").replace("=0.000", "<0.001"))
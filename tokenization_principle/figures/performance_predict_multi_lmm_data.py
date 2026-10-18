#!/usr/bin/env python3

import argparse
import collections
import glob
import os
import pickle

from multiprocess.pool import Pool
import numpy as np
from tokenization_scorer import METRIC_TITLES, score

# rsync -azP euler:/cluster/work/sachan/vilem/random-bpe/logs/train_mt_*.log logs/
# ./tokenization_principle/figures/performance_predict_multi_lmm_data.py --predictor seq_len --write-cache
# ./tokenization_principle/figures/performance_predict_multi_lmm_data.py --predictor bits --load-cache
# ./tokenization_principle/figures/performance_predict_multi_lmm_data.py --predictor renyi --power 3.0 --load-cache
# ./tokenization_principle/figures/performance_predict_multi_lmm_data.py --predictor renyi_eff --power 3.0 --load-cache
# ./tokenization_principle/figures/performance_predict_multi_lmm_data.py --predictor gowda --freq-alpha-start 0.03 --freq-alpha-end 0.83 --load-cache


# for POWER in $(seq 0.0 0.1 10.00); do
#     echo $POWER
#     ./tokenization_principle/figures/performance_predict_multi_lmm_data.py --predictor renyi_eff --freq-alpha-start 0.0 --freq-alpha-end 1.0 --power $POWER --write-cache --write-cache-name _$POWER
# done;

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--data", default="data/*/*/dev.en")
parser.add_argument("-p", "--predictor", default="bits")
parser.add_argument("--load-cache", action="store_true")
parser.add_argument("--write-cache", action="store_true")
parser.add_argument("--write-cache-name", default="")
parser.add_argument("--ci", type=float, default=0.95)
parser.add_argument("--freq-alpha-start", type=float, default=0.0)
parser.add_argument("--freq-alpha-end", type=float, default=1.0)
parser.add_argument("--power", type=float, default=None)
parser.add_argument("--central-measure", default=None)
parser.add_argument("--base-vals", default=None)
parser.add_argument("--aggregator", default=None)
parser.add_argument("--data-out", default=None)
args = parser.parse_args()
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
    with Pool() as pool:
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

with Pool() as pool:
    data_flat = pool.map(compute_metric, data_flat)

for (model_name, vocab_size_name, train_lines_name), val in data_flat:
    data[model_name][(vocab_size_name, train_lines_name)] = val

data_all = []
data_out = []
min_xs = np.inf
max_xs = -np.inf

# sort by model name
data = sorted(data.items(), key=lambda x: x[0])
if args.write_cache:
    with open(f"computed/perf_corr_cache_multi_processed{args.write_cache_name}.pkl", "wb") as f:
        pickle.dump(data, f)
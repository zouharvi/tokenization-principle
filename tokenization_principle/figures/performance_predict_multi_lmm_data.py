#!/usr/bin/env python3

import argparse
import collections
import glob
import os
import pickle

import multiprocess
import numpy as np

from tokenization_principle.figures.performance_predictors import get_predictor

# rsync -azP euler:/cluster/work/sachan/vilem/random-bpe/logs/train_mt_*.log logs/
# ./src/figures/performance_predict_multi_lmm.py --predictor seq_len --write-cache
# ./src/figures/performance_predict_multi_lmm.py --predictor bits --load-cache
# ./src/figures/performance_predict_multi_lmm.py --predictor renyi --power 3.0 --load-cache
# ./src/figures/performance_predict_multi_lmm.py --predictor renyi_eff --power 3.0 --load-cache
# ./src/figures/performance_predict_multi_lmm.py --predictor freq --freq-alpha-start 0.03 --freq-alpha-end 0.83 --power 1 --load-cache


# for POWER in $(seq 0.0 0.1 10.00); do
#     echo $POWER
#     ./src/figures/performance_predict_multi_lmm_data.py --predictor renyi_eff --freq-alpha-start 0.0 --freq-alpha-end 1.0 --power $POWER --write-cache --write-cache-name _$POWER
# done;

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

predictor, predictor_title = get_predictor(args.predictor)

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


with multiprocess.Pool() as pool:
    data_flat = pool.map(
        lambda x: (
            # signature stuff
            x[0], (
                # data, vocab_size, args
                predictor(x[1][0][0], x[1][0][1], args_kwargs),
                # perf
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
if args.write_cache:
    with open(f"computed/perf_corr_cache_multi_processed{args.write_cache_name}.pkl", "wb") as f:
        pickle.dump(data, f)
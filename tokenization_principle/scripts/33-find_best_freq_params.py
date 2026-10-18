#!/usr/bin/env python3

import argparse
import json
import pickle
from itertools import product

import multiprocess
import numpy as np
import tqdm
from scipy.stats import pearsonr, spearmanr
from sklearn.model_selection import train_test_split
from tokenization_scorer.metrics import get_prob_distribution

args = argparse.ArgumentParser(description="Find optimal start and end alpha parameters for percentile frequency (Gowda).")
args.add_argument("-d", "--data", default="data/model_bpe_random/*/dev.en")
args.add_argument("--cache", default="computed/perf_corr_cache.pkl")
args.add_argument("--step", type=float, default=0.01)
args = args.parse_args()

step_count = int(1.0 / args.step)
freq_alphas = np.linspace(0, 1, step_count)
freq_alphas = [(s, e) for s, e in product(freq_alphas, freq_alphas) if s <= e]

with open(args.cache, "rb") as f:
    data_flat = pickle.load(f)

data_processed = []
data_flat_perfs = []
for line in tqdm.tqdm(data_flat, desc="Processing text"):
    (vocab_size_name, temperature), ((data, vocab_size), perf) = line
    words_freqs, probs, _ = get_prob_distribution([l.split() for l in data.splitlines() if l.strip()])
    data_processed.append((vocab_size_name, temperature, words_freqs, probs, vocab_size))
    data_flat_perfs.append(perf)


def predictor_gowda(probs, start_alpha, end_alpha):
    start_i = min(int(len(probs) * start_alpha), len(probs) - 1)
    end_i = min(int(len(probs) * end_alpha), len(probs) - 1)
    if start_i == end_i:
        start_i = max(0, start_i - 1)
    if start_i == end_i:
        end_i = min(len(probs) - 1, end_i + 1)
    return float(np.sum([probs[i] for i in range(start_i, end_i)]))


def evaluate_pair(pair):
    start_alpha, end_alpha = pair
    data_local = [
        predictor_gowda(probs, start_alpha, end_alpha)
        for (_, _, _, probs, _) in data_processed
    ]

    data_train_x, _data_dev_x, data_train_y, _data_dev_y = train_test_split(
        data_flat_perfs, data_local, test_size=0.5, random_state=0
    )
    p_rho, p_pval = pearsonr(data_train_x, data_train_y)
    s_rho, s_pval = spearmanr(data_train_x, data_train_y)

    return {
        "start_alpha": float(start_alpha),
        "end_alpha": float(end_alpha),
        "pearson": float(p_rho),
        "pearson_p": float(p_pval),
        "spearman": float(s_rho),
        "spearman_p": float(s_pval),
    }


def main():
    with multiprocess.Pool() as pool:
        results = list(pool.imap_unordered(evaluate_pair, freq_alphas, chunksize=100))

    valid_results = [r for r in results if not np.isnan(r["pearson"])]
    best_pearson = max(valid_results, key=lambda r: abs(r["pearson"]))
    best_spearman = max(valid_results, key=lambda r: abs(r["spearman"]))

    print(f"Best Pearson: {best_pearson['pearson']:.3f} at start={best_pearson['start_alpha']:.2f}, end={best_pearson['end_alpha']:.2f}")
    print(f"Best Spearman: {best_spearman['spearman']:.3f} at start={best_spearman['start_alpha']:.2f}, end={best_spearman['end_alpha']:.2f}")

    with open("computed/best_freq_params.json", "w") as f:
        json.dump({"best_pearson": best_pearson, "best_spearman": best_spearman}, f, indent=2)


if __name__ == "__main__":
    main()

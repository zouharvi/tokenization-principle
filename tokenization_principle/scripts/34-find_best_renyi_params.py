#!/usr/bin/env python3

import argparse
import json
import pickle

import multiprocess
import numpy as np
import tqdm
from scipy.stats import pearsonr, spearmanr
from sklearn.model_selection import train_test_split
from tokenization_scorer.metrics import get_metric, get_prob_distribution

args = argparse.ArgumentParser(description="Find optimal power/alpha parameter for Rényi efficiency.")
args.add_argument("-d", "--data", default="data/model_bpe_random/*/dev.en")
args.add_argument("--cache", default="computed/perf_corr_cache.pkl")
args.add_argument("--power-min", type=float, default=0.1)
args.add_argument("--power-max", type=float, default=10.0)
args.add_argument("--power-step", type=float, default=0.1)
args = args.parse_args()

powers = np.arange(args.power_min, args.power_max + args.power_step / 2, args.power_step)

with open(args.cache, "rb") as f:
    data_flat = pickle.load(f)

data_processed = []
data_flat_perfs = []
renyi_eff_fn = get_metric("renyi_eff")

for line in tqdm.tqdm(data_flat, desc="Processing text"):
    (vocab_size_name, temperature), ((data, vocab_size), perf) = line
    lines = [l.split() for l in data.splitlines() if l.strip()]
    data_processed.append((vocab_size_name, temperature, lines, vocab_size))
    data_flat_perfs.append(perf)


def evaluate_power(power):
    data_local = [
        float(renyi_eff_fn(lines, vocab=vocab_size, power=float(power)))
        for (_, _, lines, vocab_size) in data_processed
    ]

    data_train_x, _data_dev_x, data_train_y, _data_dev_y = train_test_split(
        data_flat_perfs, data_local, test_size=0.5, random_state=0
    )
    p_rho, p_pval = pearsonr(data_train_x, data_train_y)
    s_rho, s_pval = spearmanr(data_train_x, data_train_y)

    return {
        "power": float(power),
        "pearson": float(p_rho),
        "pearson_p": float(p_pval),
        "spearman": float(s_rho),
        "spearman_p": float(s_pval),
    }


def main():
    with multiprocess.Pool() as pool:
        results = list(pool.imap_unordered(evaluate_power, powers))

    valid_results = [r for r in results if not np.isnan(r["pearson"])]
    best_pearson = max(valid_results, key=lambda r: abs(r["pearson"]))
    best_spearman = max(valid_results, key=lambda r: abs(r["spearman"]))

    print(f"Best Pearson: {best_pearson['pearson']:.3f} at power={best_pearson['power']:.1f}")
    print(f"Best Spearman: {best_spearman['spearman']:.3f} at power={best_spearman['power']:.1f}")

    with open("computed/best_renyi_params.json", "w") as f:
        json.dump({"best_pearson": best_pearson, "best_spearman": best_spearman, "all_results": valid_results}, f, indent=2)


if __name__ == "__main__":
    main()

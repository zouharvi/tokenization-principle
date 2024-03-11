#!/usr/bin/env python3

import json
import os
import pickle
import sys
import multiprocess
import tqdm
sys.path.append("src")
import argparse
import collections
import numpy as np
import scipy
from scipy.stats import pearsonr, spearmanr
from itertools import product
from figures.predictors_bleu import get_prob_distribution

# rsync -azP euler:/cluster/work/sachan/vilem/random-bpe/logs/train_mt_*.log logs/
# ./src/figures/predict_bleu.py --predictor entropy --write-cache
# ./src/figures/predict_bleu.py --predictor seq_len
# ./src/figures/predict_bleu.py --predictor renyi_eff --power 3.0 --load-cache
# ./src/figures/predict_bleu.py --predictor renyi --power 3.0 --load-cache
# ./src/figures/predict_bleu.py --predictor freq --freq-alpha-start 0.80 --freq-alpha-end 0.90 --power 1 --load-cache
# ./src/figures/predict_bleu.py --predictor freq_prob --freq-alpha-start 0.75 --freq-alpha-end 0.90 --power 1 --load-cache
# ./src/figures/predict_bleu.py --predictor freq_prob_square --freq-alpha-start 0.25 --freq-alpha-end 0.75 --load-cache


args = argparse.ArgumentParser()
args.add_argument("-d", "--data", default="data/model_bpe_random/*/dev.en")
args = args.parse_args()
args_kwargs = vars(args)

freq_alphas = np.arange(0, 1000, 1)/1000
freq_alphas = product(freq_alphas, freq_alphas)

def predictor_renyi_eff(probs, vocab_size, power):
    if power == 1:
        scale = 1
    else:
        scale = 1/(1-power)

    freqs = scale*np.log2(np.sum([
        (probs[index])**power
        for index in range(len(probs))
    ]))/np.log2(vocab_size)
    return freqs

data_flat = pickle.load(open("computed/bleu_corr_cache.pkl", "rb"))
data_flat_new = []
data_flat_renyis = []
for line in tqdm.tqdm(data_flat):
    (vocab_size_name, temperature), ((data, vocab_size), bleu) = line
    words_freqs, probs = get_prob_distribution(data)
    data_flat_new.append(
        (vocab_size_name, temperature, words_freqs, probs, vocab_size)
    )

    renyi_value = predictor_renyi_eff(probs, vocab_size, 3.0)
    data_flat_renyis.append(renyi_value)

data_flat = data_flat_new


def predictor_freq_prob(probs, vocab_size, freq_alphas):
    freq_alpha_start, freq_alpha_end = freq_alphas

    start_i = min(
        int(len(probs) * freq_alpha_start),
        len(probs) - 1
    )
    end_i = min(
        int(len(probs) * freq_alpha_end), len(probs) - 1
    )
    if start_i == end_i:
        start_i = max(0, start_i-1)
    if start_i == end_i:
        end_i = min(len(probs)-1, end_i+1)

    indicies = range(start_i, end_i)

    # indicies = [int(x) for x in np.linspace(
    #     start_i, end_i + 0.001, 10
    # )]

    freqs = np.sum([
        probs[i]
        for i in indicies
    ])
    # TODO: try to remove the log2 division
    #  / np.log2(vocab_size)
    return freqs


def tasker(freq_alphas):
    if freq_alphas[0] > freq_alphas[1]:
        return None
    data_local = [
        predictor_freq_prob(words_freqs, vocab_size, freq_alphas)
        for (vocab_size_name, temperature, words_freqs, probs, vocab_size) in data_flat
    ]
    pearson_rho, pearson_pval = pearsonr(data_flat_renyis, data_local)
    spearman_rho, spearman_pval = spearmanr(data_flat_renyis, data_local)

    return {
        "pearson": pearson_rho, "spearman": spearman_rho,
        "pearson_p": pearson_pval, "spearman_p": spearman_pval,
        "start_alpha": freq_alphas[0],
        "end_alpha": freq_alphas[1],
    }

with multiprocess.Pool() as pool:
    data_out = pool.map(
        tasker,
        tqdm.tqdm(list(freq_alphas))
    )

for line in data_out:
    if line is not None:
        print(json.dumps(line))
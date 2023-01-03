#!/usr/bin/env python3

import collections
import argparse
import random
import tqdm

args = argparse.ArgumentParser()
args.add_argument(
    "-di", "--data-in", nargs="+",
    default=[
        "data/CCrawl.de-en/dev.tok.de",
        "data/CCrawl.de-en/dev.tok.en",
    ])
args.add_argument(
    "-do", "--data-out", nargs="+",
    default=[
        "data/flattened/n3_e10_p01_k3/dev.de",
        "data/flattened/n3_e10_p01_k3/dev.en",
    ])
args.add_argument("-n", type=int, default=2)
args = args.parse_args()


def fit_ngram_char_model(datas):
    model = collections.defaultdict(lambda: collections.defaultdict(int))
    word_freqs = list(collections.Counter([
        word
        for data in datas
        for line in data
        for word in line
    ]).most_common())
    for word, freq in word_freqs:
        word_new = " " * args.n + word + " " * args.n
        for i in range(args.n, len(word) + args.n):
            skip_gram = word_new[i - args.n:i], word_new[1+i + args.n:i]
            model[skip_gram][word_new[i]] += freq

    # flatten to top k (10)
    model = {
        k: sorted(v.items(), key=lambda x: x[1], reverse=True)[:10]
        for k, v in model.items()
    }
    model = {
        k: (
            [c for c, freq in v],
            [freq for c, freq in v],
        )
        for k, v in model.items()
    }
    return model


def apply_ngram_char_model(datas, model):
    vocab = {
        word
        for data in datas
        for line in data
        for word in line
    }
    vocab = list(vocab)
    print(len(vocab), "vocab size")

    char_total = 0
    char_not_hit = 0
    # edit only word ids to make this faster
    replacement_vocab = dict()
    for word in vocab:
        modified = False
        word_new = " " * args.n + word + " " * args.n
        word_builder = ""
        for i in range(args.n, len(word) + args.n):
            observed_char = word_new[i]
            skip_gram = word_new[i - args.n:i], word_new[1+i + args.n:i]
            possible_char, possible_char_weights = model[skip_gram]
            char_total += 1
            if observed_char in possible_char:
                word_builder += observed_char
            else:
                modified = True
                word_builder += random.choices(
                    population=possible_char,
                    weights=possible_char_weights,
                    k=1
                )[0]
                char_not_hit += 1

        replacement_vocab[word] = word_builder

    print(f"Not hit percentage {char_not_hit/char_total:.2%}")

    datas = [
        [
            [
                replacement_vocab[word]
                for word in line
            ]
            for line in data
        ]
        for data in datas
    ]

    return datas, char_not_hit/char_total


datas = [
    [
        line.rstrip("\n").split(" ")
        for line in open(file, "r").readlines()[:10000]
    ]
    for file in args.data_in
]

while True:
    print("fitting")
    model = fit_ngram_char_model(datas)
    print("applying")
    datas, modified_perc = apply_ngram_char_model(datas, model)
    if modified_perc < 0.02:
        break

for fname, data in zip(args.data_out, datas):
    with open(fname, "w") as f:
        f.write("\n".join([" ".join(line) for line in data]))

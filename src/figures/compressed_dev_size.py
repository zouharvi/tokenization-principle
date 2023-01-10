#!/usr/bin/env python3

import collections

import numpy as np
import fig_utils
import matplotlib.pyplot as plt
import argparse
import json

# scp euler:/cluster/work/sachan/vilem/random-bpe/computed/paten.jsonl computed/

args = argparse.ArgumentParser()
args.add_argument("-l", "--logfile", default="computed/paten.jsonl")
args = args.parse_args()

with open(args.logfile, "r") as f:
    data = [json.loads(x) for x in f.readlines()]

data_en = [x for x in data if "dev.en" in x["output"]]
data_de = [x for x in data if "dev.de" in x["output"]]

VOCAB_COLORS = {
    4000: fig_utils.COLORS[0],
    8000: fig_utils.COLORS[1],
    16000: fig_utils.COLORS[2],
}

def get_vocab_size(output):
    return int(output.split("/")[-2].split("_")[-1].replace("k", "000"))

MODEL_I = {}
VOCAB_I = {}

for line_i, line_en in enumerate(data_en):
    if "/" in line_en["model"]:
        model_name = line_en["method"]
    else:
        model_name = line_en["model"]+"/"+line_en["method"]
    vocab_size = get_vocab_size(line_en["output"])

    if model_name == "morfessor" and vocab_size == 8000:
        if line_en["output"].split("/")[-2].split("_")[0] == "l2k":
            continue
            # print(model_name, vocab_size, )

    vocab_present = True
    if vocab_size not in VOCAB_I:
        VOCAB_I[vocab_size] = max([-1] + list(VOCAB_I.values()))+1
        vocab_present = False

    if model_name not in MODEL_I:
        MODEL_I[model_name] = max([-1] + list(MODEL_I.values()))+1

    model_i = MODEL_I[model_name]

    line_de = [
        line
        for line in data_de
        if line["output"].removesuffix(".de") == line_en["output"].removesuffix(".en")
    ][0]
    data_value = [line_en["total_subwords"]+line_de["total_subwords"]]
    plt.scatter(
        [data_value], [model_i+VOCAB_I[vocab_size]/10],
        color=VOCAB_COLORS[vocab_size],
        label=None if vocab_present else f"{vocab_size//1000}k"
    )

YTICKS = sorted(MODEL_I.items(), key=lambda x: x[1])
plt.yticks(
    range(len(YTICKS)),
    [x[0] for x in YTICKS]
)

plt.xlabel("Subword count of 100k dev lines")
plt.legend()
plt.tight_layout()
plt.show()

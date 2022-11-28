#!/usr/bin/env python3

import collections
import fig_utils
import matplotlib.pyplot as plt
import argparse
import json

args = argparse.ArgumentParser()
args.add_argument("-l", "--logfile", default="computed/apply_bpe_all_bleu.jsonl")
args = args.parse_args()

with open(args.logfile, "r") as f:
    data = [json.loads(x) for x in f.readlines()]

data_vocabsize = collections.defaultdict(dict)
for line_train in data:
    model = line_train["model"]
    vocab_size = model.split("_")[-1]
    data_vocabsize[vocab_size][model]=line_train

POINT_COLOR = {
    "4096": fig_utils.COLORS[0],
    "8192": fig_utils.COLORS[1],
    "16384": fig_utils.COLORS[2],
}
MARKER_STYLE = {
    "greedy": ".",
    "antigreedy": "v",
    "random": "*"
}

def get_quantity(line):
    return line["bleu"]

for vocab_size, data_vocabsize_local in data_vocabsize.items():
    for model, line_train in data_vocabsize_local.items():
        q_yaxis = get_quantity(line_train)
        q_xaxis = line_train["total_subwords"]/1000000

        model_name = model.split("_")[0]
        name = model

        if "random" in name:
            if any([f"_{x}_" in name for x in [1, 2, 3, 4]]):
                name = None
            else:
                name = name.split("_")
                name = name[0] + "_" + name[2]

        if name is not None:
            name = name.replace("_", " ")

        plt.scatter(
            q_xaxis, q_yaxis,
            label=name,
            color=POINT_COLOR[vocab_size],
            marker=MARKER_STYLE[model_name]
        )

plt.ylabel(r"BLEU (avg EN$\leftrightarrow$DE)")
plt.xlabel("Subword count of 200k train lines (millions)")
plt.legend()
plt.tight_layout()
plt.show()
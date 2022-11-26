#!/usr/bin/env python3

import collections
import fig_utils
import matplotlib.pyplot as plt
import argparse
import json

args = argparse.ArgumentParser()
args.add_argument("-l", "--logfile", default="computed/apply_bpe_all.jsonl")
args.add_argument("-y", "--y-axis", default="entropy_subword")
args = args.parse_args()


with open(args.logfile, "r") as f:
    data = [json.loads(x) for x in f.readlines()]

data_train = [x for x in data if "train" in x["input"]]
data_dev = [x for x in data if "dev" in x["input"]]
data = [
    (x, [y for y in data_dev if y["model"] == x["model"]][0])
    for x in data_train
]

data_vocabsize = collections.defaultdict(lambda: collections.defaultdict(list))
for line_train, line_dev in data:
    model = line_train["model"].removesuffix(".bpe_model")
    vocab_size = model.split("_")[-1]
    data_vocabsize[vocab_size][model].append((line_train, line_dev))

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

def get_quantity(line_train, line_dev):
    if args.y_axis == "dev_subwords":
        return line_dev["total_subwords"]/1000000
    elif args.y_axis == "entropy_subword":
        return line_train["entropy_subword"]
    else:
        raise Exception("Unknown quantity")
    

for vocab_size, data_vocabsize_local in data_vocabsize.items():
    for model, data_vocabsize_local_local in data_vocabsize_local.items():
        line_train, line_dev = data_vocabsize_local_local[0]
        line_train_1, line_dev_1 = data_vocabsize_local_local[1]
        q_yaxis_0 = get_quantity(line_train, line_dev)
        q_yaxis_1 = get_quantity(line_train_1, line_dev_1)
        q_yaxis = (q_yaxis_0+q_yaxis_1)/2
        q_xaxis = (line_train["total_subwords"]/1000000+line_train_1["total_subwords"]/1000000)

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

plt.ylabel(r"Subword entropy $-\frac{1}{|C|}\sum_{sw.} \log_2 p(sw.)$")
plt.xlabel("Subword count of 200k train lines (M)")
plt.legend()
plt.tight_layout()
plt.show()
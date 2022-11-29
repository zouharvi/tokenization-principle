#!/usr/bin/env python3

import collections
import fig_utils
import matplotlib.pyplot as plt
import argparse
import json

args = argparse.ArgumentParser()
args.add_argument("-l", "--logfile", default="computed/applybpe_small.jsonl")
args = args.parse_args()

with open(args.logfile, "r") as f:
    data = [json.loads(x) for x in f.readlines()]

data_dev = [x for x in data if "dev" in x["input"]]
data_train = [x for x in data if "train" in x["input"]]

data_new = []
for line_dev in data_dev:
    # process only English ones
    # we process both but lets use the En as the main
    if line_dev["input"].endswith(".de"):
        continue

    line_train_all = [
        x for x in data_train
        if x["model"] == line_dev["model"] and x["method"] == line_dev["method"]
    ]
    line_dev_all = [
        x for x in data_dev
        if x["model"] == line_dev["model"] and x["method"] == line_dev["method"]
    ]

    assert len(line_train_all) == 2
    assert len(line_dev_all) == 2

    line_new = {
        "train_subwords": (line_train_all[0]["total_subwords"]+line_train_all[1]["total_subwords"])/2,
        "dev_subwords": (line_dev_all[0]["total_subwords"]+line_dev_all[1]["total_subwords"])/2,
        "method": line_dev["method"],
        "model": line_dev["model"].rstrip(".bpe_merges"),
    }
    data_new.append(line_new)

POINT_COLOR = {
    "greedy": fig_utils.COLORS[1],
    "greedyalmost": fig_utils.COLORS[4],
    "random uni": fig_utils.COLORS[3],
    "random softmax": fig_utils.COLORS[0],
    "antigreedy": fig_utils.COLORS[2],
}
MARKER_STYLE = {
    "greedy_naive": "v",
    "merge_operations": "*",
}

plt.figure(figsize=(8,6))

displayed_labels = set()
displayed_labels_full = set()

for line in data_new:
    name = line["model"].replace("random_", "random")
    name = name.split("_")[0]
    name = name.replace("random", "random ")
    name_all = name + " / " + line["method"]
    name_all = name_all.replace("merge_operations", "merge").replace("greedy_naive", "greedy length")
    name_all = name_all.replace("random uni", "random unigram")

    if "random_softmax_t" in line["model"]:
        extra_label = "" + line["model"].split("_")[2].removeprefix("t").replace("0", "0.")
    elif "greedyalmost_" in line["model"]:
        extra_label = "" + line["model"].split("_")[1]
    else:
        extra_label = ""

    name_all_full = name_all + extra_label
    if name_all_full in displayed_labels_full:
        continue
    displayed_labels_full.add(name_all_full)


    plt.text(
        line["train_subwords"]/1000, line["dev_subwords"]/1000+(20 if line["method"] == "greedy_naive" else -20),
        s=extra_label, ha="center", va="center",
    )

    plt.scatter(
        line["train_subwords"]/1000, line["dev_subwords"]/1000,
        label=name_all if name_all not in displayed_labels else None,
        color=POINT_COLOR[name],
        marker=MARKER_STYLE[line["method"]]
    )
    displayed_labels.add(name_all)

plt.suptitle("Numbers next to point indicate parameter (either temperature for softmax or\nn for almostgreedy). Legend labels are as 'trainig method'/'decoding method'.")
plt.xlabel("Subword count of 20k train lines (in thousands)")
plt.ylabel("Subword count of 20k train lines (in thousands)")
plt.legend(bbox_to_anchor=(0, 1, 1, 0), loc="lower left", mode="expand", ncol=2)
plt.tight_layout()
plt.show()
#!/usr/bin/env python3

import tabulate
import json
import collections
import argparse

args = argparse.ArgumentParser()
args.add_argument("--data", default="computed/all_predictors_grid.jsonl")
args.add_argument("--metric", default="pearson")
args = args.parse_args()

data = [json.loads(x) for x in open(args.data, "r") if len(x) > 1]

data_groupped = collections.defaultdict(list)

for line in data:
    data_groupped[line["args"]["predictor"]].append(line)

data_groupped = {
    k: (
        max(v, key=lambda line: abs(line["pearson"])),
        max(v, key=lambda line: abs(line["spearman"]))
    )
    for k, v in data_groupped.items()
}


def filter_args(args):
    return {
        k: v
        for k, v in args.items()
        if k not in {"data", "predictor", "load_cache", "write_cache", "ci", "no_graphics"} and v is not None
    }


print(tabulate.tabulate(
    [
        (
            name,
            f'{line_p["pearson"]:.1%}',
            f'{line_s["spearman"]:.1%}',
            filter_args(line_p["args"]),
            filter_args(line_s["args"]),
        )
        for name, (line_p, line_s) in data_groupped.items()
    ],
    headers=["Name", "Max Pearson", "Max Spearman",
             "Args Pearson", "Args Spearman"],
))

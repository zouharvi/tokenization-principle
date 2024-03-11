#!/usr/bin/env python3

import argparse
import collections
import json
import re
import glob

import numpy as np

args = argparse.ArgumentParser()
args.add_argument("--glob", default="logs/train_mt_*.log")
args.add_argument("--input-logfile", default="computed/apply_bpe_all.jsonl")
args.add_argument("--output-logfile", default="computed/apply_bpe_all_bleu.jsonl")
args = args.parse_args()

re_epoch = re.compile(r" epoch (\d+)")
re_best_bleu = re.compile(r" best_bleu (\d+\.?\d*)")
re_bleu = re.compile(r" bleu (\d+\.?\d*)")

with open(args.input_logfile, "r") as f:
    data_logfile = [json.loads(x) for x in f.readlines()]

data_raw = collections.defaultdict(list)
for fname in glob.glob(args.glob):
    with open(fname, "r") as f:
        data_local = [
            x.rstrip()
            for x in f.readlines()
            if "best_bleu" in x
        ]
    if len(data_local) == 0:
        continue

    data_local = [
        (
            int(re_epoch.search(x).group(1)),
            float(re_best_bleu.search(x).group(1)),
            float(re_bleu.search(x).group(1)),
        )
        for x in data_local
    ]
    # max should take the first max best_bleu
    best_bleu = max(data_local, key=lambda x: x[1])
    cur_epoch = max(data_local, key=lambda x: x[0])
    model_name = fname.split("/")[-1]
    model_name = model_name.removeprefix("train_mt_").removesuffix(".log")
    model_name, lang = model_name.split(".")
    data_raw[model_name].append(best_bleu)

data = {}
for model_name, data_local in data_raw.items():
    avg_bleu = np.average([x[1] for x in data_local])
    print(f"{model_name:>20}: {avg_bleu:.2f}")
    data_logfile_local = [x for x in data_logfile if x["model"].startswith(model_name) and "train" in x["output"]]
    
    assert len(data_logfile_local) == 2

    data[model_name] = {
        "model": model_name,
        "bleu": avg_bleu,
        "total_subwords": data_logfile_local[0]["total_subwords"]+data_logfile_local[1]["total_subwords"]
    }

with open(args.output_logfile, "w") as f:
    f.write("\n".join(json.dumps(x) for x in data.values()))
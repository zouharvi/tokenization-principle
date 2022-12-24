#!/usr/bin/env python3

import argparse
import json
import subprocess
import tempfile

args = argparse.ArgumentParser()
args.add_argument(
    "-i", "--input", nargs="+",
    default=[
        "data/CCrawl.de-en/train.tok.en",
        "data/CCrawl.de-en/train.tok.de"
    ]
)
args.add_argument(
    "-vo", "--vocab-output",
    default="data/model_morfessor/model.pkl"
)
args.add_argument("-vs", "--vocab-size", type=int, default=8192)
args.add_argument("-n", "--number-of-lines", type=int, default=100000)
# only recursive works
# args.add_argument("--algorithm", default="recursive")
# morfessor, flatcat
args.add_argument("--morfessor", default="morfessor")
args = args.parse_args()

with tempfile.NamedTemporaryFile() as fname:
    fname = fname.name
    with open(fname, "w") as f:
        for input_fname in args.input:
            data = open(input_fname, "r").readlines()[:args.number_of_lines]
            # data = [line.replace(" ", "\n") for line in data]
            f.writelines(data)

    txt_train_command = f"\
        {args.morfessor}-train \
        {fname} \
        -s {args.vocab_output}\
    "

    if args.morfessor != "flatcat":
        txt_train_command += f" --num-morph-types {args.vocab_size} "
    else:
        pass
        # txt_train_command += f" -W {args.W} "
        # txt_train_command += f" -p {args.p} "

    print("RUNNING", txt_train_command)
    subprocess.run(txt_train_command, shell=True)
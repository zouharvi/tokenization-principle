#!/usr/bin/env python3

from lzw import BaseLZW
import argparse

args = argparse.ArgumentParser()
args.add_argument(
    "-i", "--input", nargs="+",
    default=[
        "data/CCrawl.de-en/train.tok.en",
        "data/CCrawl.de-en/train.tok.de"
    ]
)
args.add_argument("-vo", "--vocab-output", default="data/model_lzw/model.vocab")
args.add_argument("-vs", "--vocab-size", type=int, default=8192)
args.add_argument("-n", "--number-of-lines", type=int, default=100000)
args = args.parse_args()

print("Loading data")
datas = []
for f in args.input:
    with open(f, "r") as f:
        data_local = [
            [w for w in x.rstrip("\n").split(" ")] + ["\n"]
            for x in f.readlines()[:args.number_of_lines]
        ]
        datas.append(data_local)

data = [val for tup in zip(*datas) for val in tup]
# flatten
data = [w for line in data for w in line]
print(data[:10])

model = BaseLZW()
data = model.fit(data, V=args.vocab_size)
print(data[:10])

model.save(args.vocab_output)

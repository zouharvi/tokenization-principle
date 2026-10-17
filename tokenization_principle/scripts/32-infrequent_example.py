#!/usr/bin/env python3

import argparse
import collections

args = argparse.ArgumentParser()
args.add_argument("-d", "--data", default="data/de-en/train.tok.en")
args = args.parse_args()

print("Loading data")
with open(args.data, "r") as f:
    data = f.read().split()
print("Filtering data")
# take only alpha words
data = [w for w in data if w.isalpha() and w.islower() and len(w) < 20]
print("Processing data")
freqs = collections.Counter(data)
print("Printing data")
print([w for w,f in freqs.most_common() if f == 1])
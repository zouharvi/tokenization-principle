#!/usr/bin/env python3

import sys
sys.path.append("src")
import argparse
import collections

args = argparse.ArgumentParser()
args.add_argument("-d", "--data", default="data/CCrawl.de-en/train.tok.en")
args = args.parse_args()

print("Loading data")
data = open(args.data, "r").read().split()
print("Filtering data")
# take only alpha words
data = [w for w in data if w.isalpha() and w.islower() and len(w) < 20]
print("Processing data")
freqs = collections.Counter(data)
print("Printing data")
print([w for w,f in freqs.most_common() if f == 1])
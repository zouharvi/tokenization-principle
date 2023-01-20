#!/usr/bin/env python3

import fig_utils
import matplotlib.pyplot as plt
import argparse
import numpy as np
import collections
import re

args = argparse.ArgumentParser()
args.add_argument("-f", type=int, default=0)
args.add_argument("-d", "--data", default="data/CCrawl.de-en/dev.en")
args = args.parse_args()

plt.figure(figsize=(2, 1.2))

# data = open(args.data, "r").read().lower()
# data = re.sub(r'[^A-Za-z]+', '', data)
# freqs = dict(collections.Counter(data))
# special_subword_freq = data.count(special_subword)
if args.f == 0:
    special_subword = "the"
    special_subword_freq = 110
elif args.f == 1:
    special_subword = "cow"
    special_subword_freq = 1
# print(sorted(freqs.items(),reverse=True, key=lambda x: x[1]))
freqs = {
    ' ': 450,
    'e': 317,
    't': 228,
    'a': 220,
    'o': 219,
    'i': 202,
    'n': 190,
    'r': 187,
    's': 178,
    'l': 116,
    'c': 102,
    'd': 95,
    'h': 114,
    'u': 87,
    'm': 73,
    'p': 69,
    'f': 61,
    'g': 54,
    'y': 50,
    'w': 43,
    'b': 40,
    'v': 31,
    'k': 20,
    'x': 7,
    'q': 4,
    'j': 4,
    'z': 4
}

for c in special_subword:
    freqs[c] -= special_subword_freq
freqs[special_subword] = special_subword_freq
COLORS = [fig_utils.COLORS[0]]*len(freqs)
data_y = sorted(freqs.items(), reverse=True, key=lambda x: x[1])

for i, (k, _) in enumerate(data_y):
    if k in "cow":
        COLORS[i] = "#dc7"
    elif k in "the":
        COLORS[i] = "#8c8"

data_y = np.array([x[1] for x in data_y])
data_y = data_y/data_y.sum()

def entropy(data, base):
    data = np.array(data)
    return -np.sum(data * np.log2(data) / np.log2(base))

def renyi(data, power):
    scale = 1 / (1 - power)
    return scale * np.log2(np.sum(np.power(data, power)))

h_i = renyi(data_y, 200)
h_3 = renyi(data_y, 9.0)
h_1 = entropy(data_y, base=2)
h_0 = renyi(data_y, 0.0)

data_x = np.array(range(len(data_y)))

plt.bar(
    data_x,
    data_y,
    # edgecolor="black",
    linewidth=0,
    width=1.01,
    color=COLORS
)
plt.xticks([])
plt.yticks([])
plt.xlim(data_x.min()- 0.5, data_x.max() - 4)


plt.title(
    f"$H_0={h_0:.2f}, H_1={h_1:.2f}, H_9 ={h_3:.2f}$   \n$H_1/H_0={h_1/h_0:.0%}$%," " $H_{9}" f"/H_0={h_3/h_0:.0%}$%",
    fontsize=8.5
)
plt.ylabel("  Frequency" if args.f in {0, 2} else r" ")

plt.xlabel(
    ["More uniform", "Large peak"]
    [args.f]
)

plt.tight_layout(pad=0.5)
plt.savefig(f"computed/figures/peak_nopeak_{args.f}.pdf")
plt.show()


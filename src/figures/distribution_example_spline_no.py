#!/usr/bin/env python3

import fig_utils
import matplotlib.pyplot as plt
import argparse
import numpy as np


# for f in "0" "1" "2" "3"; do DISPLAY="" ./src/figures/distribution_example.py -f $f; done

args = argparse.ArgumentParser()
args.add_argument("-f", type=int, default=0)
args = args.parse_args()

plt.figure(figsize=(2, 1.3))

if args.f == 0:
    data_y = np.array([6, 5, 5, 5, 4, 2])
elif args.f == 1:
    data_y = np.array([6, 5, 5, 5, 1, 1])
elif args.f == 2:
    data_y = np.array([6, 5, 5, 5, 4, 2])
elif args.f == 3:
    data_y = np.array([10, 5, 5, 5, 4, 2])

# TODO: add code length for each bar

data_y = data_y / data_y.sum()
data_x = np.array(range(len(data_y)))

plt.bar(
    data_x,
    data_y,
    edgecolor="black",
    width=1,
    color=fig_utils.COLORS[0]
)

plt.ylim(0, 0.4)
plt.xlim(data_x.min()-0.5, data_x.max()+0.5)
plt.xticks([])
plt.yticks([])


def renyi_eff(data, power):
    scale = 1 / (1 - power)
    return scale * np.log2(np.sum(np.power(data, power)))


def entropy(data):
    data = np.array(data)
    return -np.sum(data * np.log2(data))


h_i = renyi_eff(data_y, 200)
h_3 = renyi_eff(data_y, 9.0)
h_1 = entropy(data_y)
h_0 = renyi_eff(data_y, 0.0)

print(h_i)

plt.title(
    f"$H_0={h_0:.2f}, H_1={h_1:.2f}, H_9 ={h_3:.2f}$   \n$H_1/H_0={h_1/h_0:.0%}$%," " $H_{9}" f"/H_0={h_3/h_0:.0%}$%",
    fontsize=8.5
)
plt.ylabel("  Frequency" if args.f in {0, 2} else r" ")

plt.xlabel(
    ["No dip", "Dip", "No peak", "Peak"]
    [args.f]
)
plt.tight_layout(pad=0.5)
plt.savefig(f"computed/figures/distribution_example_f{args.f}.pdf")
plt.show()

#!/usr/bin/env python3

import fig_utils
import matplotlib.pyplot as plt
import argparse
import numpy as np

# for f in "0" "1" "2" "3"; do DISPLAY="" ./src/figures/distribution_example_chopped.py -f $f; done

args = argparse.ArgumentParser()
args.add_argument("-f", type=int, default=0)
args = args.parse_args()

plt.figure(figsize=(2, 1.2))

if args.f == 0:
    data_y = np.array([30, 15, 14, 14, 13, 12])
elif args.f == 1:
    data_y = np.array([30, 15, 14, 14, 13, 12, 4, 4, 2, 2, 2, 2])
elif args.f == 2:
    data_y = np.array([45, 40, 20, 20, 11, 10])
elif args.f == 3:
    data_y = np.array([45, 35, 14, 14, 13, 12, 11, 11, 9, 8, 5, 3])


data_y = data_y / data_y.sum()
data_x = np.array(range(len(data_y)))

plt.bar(
    data_x,
    data_y,
    width=1,
    edgecolor="black",
    color=fig_utils.COLORS[1],
)

plt.xlim(data_x.min()-0.5, data_x.max()+0.5)
plt.ylim(0, 0.35)
plt.xticks([])
plt.yticks([])

def renyi_eff(data, power):
    scale = 1/(1-power)
    return scale*np.log2(np.sum(np.power(data, power)))

def entropy(data):
    data = np.array(data)
    return -np.sum(data * np.log2(data))

h1 = entropy(data_y)
eff_h1 = entropy(data_y)/np.log2(len(data_y))

plt.title(
    f"$|\\Delta|={len(data_y):.0f}$, " + r"$\mathrm{H}$" + f"=${h1:.2f}$, Eff=${eff_h1:.0%}$".replace("%", r"\%"),
    fontsize=8.5
)
plt.ylabel(
    "Frequency",
    color="black" if args.f == 0 else "white",
)

plt.xlabel(
    ["Efficient", "Inefficient", "Inefficient", "Mediocre"]
    [args.f]
)
plt.tight_layout(pad=0.5)
plt.savefig(f"computed/figures/distribution_example_chopped_f{args.f}.pdf")
plt.show()
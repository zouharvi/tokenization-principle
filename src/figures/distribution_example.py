#!/usr/bin/env python3

import fig_utils
import matplotlib.pyplot as plt
import argparse
import numpy as np
from scipy.interpolate import make_interp_spline

args = argparse.ArgumentParser()
args.add_argument("-f", type=int, default=0)
args = args.parse_args()

plt.figure(figsize=(2, 1))

if args.f == 0:
    data_y = np.array([12, 8, 4, 3, 1.8, 1])
elif args.f == 1:
    data_y = np.array([5.5, 4.5, 4, 3.5, 3.3, 2.5])
data_y = data_y / data_y.sum()
data_x = np.array(range(len(data_y)))
data_y_spline = make_interp_spline(data_x, data_y)
vals_x = np.linspace(data_x.min(), data_x.max(), 500)
vals_y = data_y_spline(vals_x)

plt.fill_between(
    vals_x,
    np.zeros_like(vals_x),
    vals_y,
)

plt.ylim(0, 0.44)
plt.xlim(data_x.min(), data_x.max())
plt.xticks([])
plt.yticks([])


plt.ylabel("Frequency" if args.f == 0 else r" ")
plt.xlabel("Subword")
plt.tight_layout(pad=0)
plt.savefig(f"computed/figures/distribution_example_f{args.f}.pdf")
plt.show()
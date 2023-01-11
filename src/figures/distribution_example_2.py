#!/usr/bin/env python3

import fig_utils
import matplotlib.pyplot as plt
import argparse
import numpy as np

args = argparse.ArgumentParser()
args.add_argument("-k", type=int, default=0)
args = args.parse_args()

def entropy(data, base):
    data = np.array(data)
    print("SUM", np.sum(data), np.sum(data * np.log2(data) / np.log2(base)))
    return -np.sum(data * np.log2(data) / np.log2(base))

def renyi_entropy(data, alpha):
    if alpha == 1:
        ent_val = entropy(data, len(data))
        print("falling back to entropy", ent_val)
        return ent_val
    scale = 1/(1-alpha)
    return scale*np.log2(np.sum(np.power(data, alpha)))/np.log2(len(data))

data_y = np.random.rand(50)
data_y = data_y / data_y.sum()
print(entropy(data_y, len(data_y)))
exit()


def funfun(b, k):
    global data_y

    ent_val = entropy(data_y, base=b)

    log_val = np.log2(k)/np.log2(b)
    val_out = ent_val / log_val

    return ent_val / np.log2(k)

def funfun2(alpha, k):
    data_y = np.random.rand(k)
    data_y = data_y / data_y.sum()

    ent_val = renyi_entropy(data_y, alpha=alpha)
    return ent_val

K = 50
BS = range(2, K)
vals = [
    funfun(b, K)
    for b in BS
]


ALPHAS = np.linspace(0, 9, num=10)
# plt.plot(
#     ALPHAS,
#     [funfun2(a, K) for a in ALPHAS],
#     label="renyi",
# )
plt.plot(
    BS,
    [funfun(b, K) for b in BS],
)
# plt.plot(
#     BS,
#     [v / (np.log2(K)/np.log2(b)) for v, b in zip(vals, BS)]
# )

plt.legend()
plt.tight_layout(pad=0.5)
plt.show()


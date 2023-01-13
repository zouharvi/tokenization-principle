#!/usr/bin/env python3

import random
import numpy as np
from dahuffman import HuffmanCodec
import string


data = np.random.random((50))
data.sort()
data = data[::-1]
data /= data.sum()

print(data)
data_freqs = {k:v for k,v in zip(string.ascii_letters, data)}

def entropy(data):
    data = np.array(data)
    return -np.sum(data * np.log2(data))

def renyi_entropy(data, alpha):
    if alpha == 1:
        ent_val = entropy(data, len(data))
        return ent_val
    scale = 1/(1-alpha)
    return scale*np.log2(np.sum(np.power(data, alpha)))

def cost_fun(data, data_lens, t):
    return 1/t * np.log2(np.sum(data * np.power(2, t * data_lens)))

CODETYPE="random_huff"
if CODETYPE == "huff":
    codec = HuffmanCodec.from_frequencies(data_freqs)
    data_lens = [l-1 for k, (l, code) in codec.get_code_table().items() if type(k) is str]
elif CODETYPE == "random_huff":
    codec = HuffmanCodec.from_frequencies(data_freqs)
    data_lens = [l-1 for k, (l, code) in codec.get_code_table().items() if type(k) is str]
    random.shuffle(data_lens)
elif CODETYPE == "const_onehot":
    data_lens = [len(data)]*len(data)
elif CODETYPE == "const_loglen":
    data_lens = [np.ceil(np.log2(len(data)))]* len(data)
elif CODETYPE == "const_entropy":
    data_lens = [np.ceil(entropy(data))]* len(data)


data_lens = np.array(data_lens)
print(data_lens)
assert len(data_lens) == len(data)


for t in np.linspace(-1, 5, 10):
    if t in {-1, 0}:
        continue
    alpha = (1+t)**(-1)
    ent_value = renyi_entropy(data, alpha)
    cost_value = cost_fun(data, data_lens, t)

    print(f"T={t:.2f} a={alpha:.2f} H={ent_value:.2f}, L={cost_value:.2f}")

print()

for alpha in np.linspace(0, 5, 10):
    if alpha == 0:
        continue
    t = 1/alpha - 1
    ent_value = renyi_entropy(data, alpha)
    cost_value = cost_fun(data, data_lens, t)

    print(f"T={t:.2f} a={alpha:.2f} H={ent_value:.2f}, L={cost_value:.2f}")

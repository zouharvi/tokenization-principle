from .base import BaseBPE
import numpy as np
import random

class RandomBPE(BaseBPE):
    def __init__(self, seed, randomness_dist, randomness_temp, **kwargs):
        self.random = random.Random(seed)
        if randomness_dist == "uniform":
            self.weights = lambda pair_vals: [1]*len(pair_vals)
        elif randomness_dist == "softmax":
            def tmp(pair_vals):
                pair_vals = np.array(pair_vals, dtype=np.float128)
                pair_vals_max = np.max(pair_vals)
                # divide also by the max because otherwise we get inf
                pair_vals = (pair_vals/pair_vals_max)/randomness_temp   
                return np.exp(pair_vals)/np.sum(np.exp(pair_vals))
            self.weights = tmp
        else:
            raise Exception("Unknown distribution " + randomness_dist)

    def choose_pair_to_merge(self, pairs):
        pairs = list(pairs.items())
        return self.random.choices(
            [x[0] for x in pairs],
            weights=self.weights([x[1] for x in pairs])
        )[0]
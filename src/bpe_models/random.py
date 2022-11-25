from .base import BaseBPE
import random

class RandomBPE(BaseBPE):
    def __init__(self, seed=0):
        self.random = random.Random(seed)

    def choose_pair_to_merge(self, pairs):
        return self.random.choice(list(pairs.keys()))
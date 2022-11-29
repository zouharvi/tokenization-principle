
from .base import BaseBPE

class GreedyAlmostBPE(BaseBPE):
    def __init__(self, greedy_n, **kwargs):
        self.n = greedy_n
        print("Using n of", greedy_n)

    def choose_pair_to_merge(self, pairs):
        return sorted({p: v for p, v in pairs.items()}, key=pairs.get, reverse=True)[self.n]

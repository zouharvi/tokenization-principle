
from .base import BaseBPE

class GreedyAlmostBPE(BaseBPE):
    def __init__(self, n=1, **kwargs):
        self.n = n

    def choose_pair_to_merge(self, pairs):
        return sorted({p: v for p, v in pairs.items()}, key=pairs.get, reverse=True)[self.n]

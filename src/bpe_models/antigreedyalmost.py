from .base import BaseBPE

class AntiGreedyAlmostBPE(BaseBPE):
    def __init__(self, threshold=5, **kwargs):
        self.threshold = threshold

    def choose_pair_to_merge(self, pairs):
        return max({p: v for p, v in pairs.items() if v > self.threshold}, key=pairs.get)
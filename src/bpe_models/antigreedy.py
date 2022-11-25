from .base import BaseBPE

class AntiGreedyBPE(BaseBPE):
    def choose_pair_to_merge(self, pairs):
        return min(pairs, key=pairs.get)
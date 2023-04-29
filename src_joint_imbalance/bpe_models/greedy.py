from .base import BaseBPE

class GreedyBPE(BaseBPE):
    def choose_pair_to_merge(self, pairs, singles):
        return max(pairs, key=pairs.get)
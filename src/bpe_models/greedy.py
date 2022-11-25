from .base import BaseBPE

class GreedyBPE(BaseBPE):
    def __init__(self):
        pass

    def choose_pair_to_merge(self, pairs):
        return max(pairs, key=pairs.get)
from .base import BaseBPE
import collections
import numpy as np


class GreedyPMIBPE(BaseBPE):
    def choose_pair_to_merge(self, pairs, singles):
        total_pairs = sum(pairs.values())
        total_singles = sum(singles.values())
        return max(
            pairs,
            key=lambda pair: (
                np.log2(pairs[pair] / total_pairs)
                - np.log2(singles[pair[0]] / total_singles)
                - np.log2(singles[pair[1]] / total_singles)
            )
        )
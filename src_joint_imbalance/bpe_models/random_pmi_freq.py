from .base import BaseBPE
from .greedy import GreedyBPE
from .greedy_pmi import GreedyPMIBPE
import random

class RandomPMIFreqBPE(BaseBPE):
    def __init__(self, seed, **kwargs):
        self.random = random.Random(seed)
        self.freq_model = GreedyBPE(**kwargs)
        self.pmi_model = GreedyPMIBPE(**kwargs)

    def choose_pair_to_merge(self, pairs, singles):
        if random.choice([True, False]):
            return self.freq_model.choose_pair_to_merge(pairs, singles)
        else:
            return self.pmi_model.choose_pair_to_merge(pairs, singles)
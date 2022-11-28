from .greedy import GreedyBPE
from .antigreedy import AntiGreedyBPE
from .antigreedyalmost import AntiGreedyAlmostBPE
from .greedyalmost import GreedyAlmostBPE
from .random import RandomBPE

def get_bpe_model(name):
    if name == "greedy":
        return GreedyBPE
    elif name == "greedyalmost":
        return GreedyAlmostBPE
    elif name == "antigreedy":
        return AntiGreedyBPE
    elif name == "antigreedyalmost":
        return AntiGreedyAlmostBPE
    elif name == "random":
        return RandomBPE
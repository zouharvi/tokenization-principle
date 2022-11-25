from .greedy import GreedyBPE
from .antigreedy import AntiGreedyBPE
from .random import RandomBPE

def get_bpe_model(name):
    if name == "greedy":
        return GreedyBPE
    elif name == "antigreedy":
        return AntiGreedyBPE
    elif name == "random":
        return RandomBPE
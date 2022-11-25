from .standard import StandardBPE
from .antistandard import AntiStandardBPE
from .random import RandomBPE

def get_bpe_model(name):
    if name == "standard":
        return StandardBPE
    elif name == "antistandard":
        return AntiStandardBPE
    elif name == "random":
        return RandomBPE
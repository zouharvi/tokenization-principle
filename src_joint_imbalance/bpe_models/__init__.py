
def get_model(name):
    if name == "greedy":
        from bpe_models.greedy import GreedyBPE
        return GreedyBPE
    elif name == "greedy_pmi":
        from bpe_models.greedy_pmi import GreedyPMIBPE
        return GreedyPMIBPE
    elif name == "random_pmi_freq":
        from bpe_models.random_pmi_freq import RandomPMIFreqBPE
        return RandomPMIFreqBPE
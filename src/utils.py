
def get_model(name):
    if name == "greedy_beamsearch":
        from bpe_models.greedy_beamsearch import GreedyBeamSearchBPE
        return GreedyBeamSearchBPE
    elif name == "greedy_beamsearch_old":
        from bpe_models.greedy_beamsearch_old import GreedyBeamSearchOldBPE
        return GreedyBeamSearchOldBPE
    elif name == "greedy_beamsearch_new":
        from bpe_models.greedy_beamsearch_new import GreedyBeamSearchNewBPE
        return GreedyBeamSearchNewBPE
    elif name == "greedy":
        from bpe_models.greedy import GreedyBPE
        return GreedyBPE
    elif name == "greedycapitalizationflag":
        from bpe_models.greedy_capitalzation_flag import GreedyCapitalizationFlagBPE
        return GreedyCapitalizationFlagBPE
    elif name == "greedyalmost":
        from bpe_models.greedyalmost import GreedyAlmostBPE
        return GreedyAlmostBPE
    elif name == "antigreedy":
        from bpe_models.antigreedy import AntiGreedyBPE
        return AntiGreedyBPE
    elif name == "antigreedyalmost":
        from bpe_models.antigreedyalmost import AntiGreedyAlmostBPE
        return AntiGreedyAlmostBPE
    elif name == "random":
        from bpe_models.random import RandomBPE
        return RandomBPE
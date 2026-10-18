from typing import Any


def get_model(name: str) -> Any:
    if name == "greedy_beamsearch":
        from tokenization_principle.bpe_models.greedy_beamsearch import (
            GreedyBeamSearchBPE,
        )
        return GreedyBeamSearchBPE
    elif name == "greedy_beamsearch_old":
        from tokenization_principle.bpe_models.greedy_beamsearch_old import (
            GreedyBeamSearchOldBPE,
        )
        return GreedyBeamSearchOldBPE
    elif name == "greedy_beamsearch_new":
        from tokenization_principle.bpe_models.greedy_beamsearch_new import (
            GreedyBeamSearchNewBPE,
        )
        return GreedyBeamSearchNewBPE
    elif name == "greedy":
        from tokenization_principle.bpe_models.greedy import GreedyBPE
        return GreedyBPE
    elif name == "greedycaptrick":
        from tokenization_principle.bpe_models.greedy_captrick import GreedyCaptrickBPE
        return GreedyCaptrickBPE
    elif name == "greedyalmost":
        from tokenization_principle.bpe_models.greedyalmost import GreedyAlmostBPE
        return GreedyAlmostBPE
    elif name == "antigreedy":
        from tokenization_principle.bpe_models.antigreedy import AntiGreedyBPE
        return AntiGreedyBPE
    elif name == "antigreedyalmost":
        from tokenization_principle.bpe_models.antigreedyalmost import (
            AntiGreedyAlmostBPE,
        )
        return AntiGreedyAlmostBPE
    elif name == "random":
        from tokenization_principle.bpe_models.random import RandomBPE
        return RandomBPE
    else:
        raise ValueError(f"Unknown model: {name}")
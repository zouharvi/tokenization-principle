from .base import BaseBPE
from .base import BaseBPE
from queue import PriorityQueue
import re
import tqdm
import operator
from dataclasses import dataclass, field
from typing import Any

COLLAPSE_WHITESPACE = re.compile(r"\s+")

@dataclass(order=True)
class Hypothesis():
    score: Any=field(compare=True)

    def __init__(self, merge_operations_parent, corpus_freqs_parent):
        self.merge_operations_parent = merge_operations_parent
        self.corpus_freqs_parent = corpus_freqs_parent
        self.merge_operations = None
        self.corpus_freqs = None
        self.pair_to_add = None
        self.possible_extensions = None
        self.score = 0

    def __len__(self):
        self.get_merge_operations()
        return len(self.merge_operations)

    def spawn_child(self, pair):
        child = Hypothesis(self.get_merge_operations(), self.get_corpus_freqs())
        child.pair_to_add = pair[0]
        return child

    def get_possible_extensions(self):
        # make sure we have corpus freqs
        self.get_corpus_freqs()

        if self.possible_extensions == None:
            self.possible_extensions = BaseBPE.get_pairs(self.corpus_freqs).items()

        return self.possible_extensions

    def get_corpus_freqs(self):
        if self.corpus_freqs == None:
            if self.pair_to_add == None:
                # we don't need to clone corpus_freqs
                # TODO: ??
                self.corpus_freqs = self.corpus_freqs_parent
                pass
                # this should occur only once
                # self.corpus_freqs = copy.deepcopy(self.corpus_freqs_parent)
            else:
                # this creates a new object so we're only borrowing parent's data
                self.corpus_freqs = BaseBPE.merge_vocab(self.pair_to_add, self.corpus_freqs_parent)

        return self.corpus_freqs

    def get_merge_operations(self):
        if self.merge_operations == None:
            # hopefully this is enough for clone
            # TODO: ???
            self.merge_operations = list(self.merge_operations_parent)
            if self.pair_to_add != None:
                self.merge_operations.append(self.pair_to_add[0] + " " + self.pair_to_add[1])


        return self.merge_operations

class GreedyBeamSearchNewBPE(BaseBPE):    
    def __init__(self, beam_n: int, beam_n_expand: int, **kwargs):
        self.beam_n = beam_n
        self.beam_n_expand = beam_n_expand

    def fit(self, corpus: str, vocab_size: int):

        # ====================================================
        # NOTE: this is just a boostrap code (not beam search)
        corpus_freqs = self.build_vocab_freq(corpus)

        # get initial characters
        pairs = self.get_pairs(corpus_freqs)
        # add all characters to subword vocab even if they are not merge operations
        merge_operations = list({
            s for pair in pairs.keys() for s in pair
        })
        # add end characters
        merge_operations += list({
            pair[0] + pair[1]
            for pair in pairs.keys() if pair[1] == "</w>"
        })
        # ====================================================

        beam_t = {0: [Hypothesis(merge_operations, corpus_freqs)]}

        for t in tqdm.tqdm(range(1, vocab_size-len(merge_operations)+1)):
            beam = PriorityQueue(maxsize=0)
            for hyp in beam_t[t-1]:
                # optimization to become linear (not sure why)
                possible_pairs = list(hyp.get_possible_extensions())
                possible_pairs.sort(key=operator.itemgetter(1), reverse=True)

                for pair in possible_pairs[:self.beam_n]:
                    new_hyp = hyp.spawn_child(pair)
                    new_score = hyp.score - pair[1]
                    new_hyp.score = new_score
                    beam.put(new_hyp)

            beam_t[t] = []
            for _ in range(self.beam_n):
                beam_t[t].append(beam.get())
            # we don't need to store the history up until (exclusive) t-1
            del beam_t[t-1]

        
        beam_best = min(beam_t[t])
        
        self.merge_operations = beam_best.get_merge_operations()
        self.corpus_freqs = beam_best.get_corpus_freqs()

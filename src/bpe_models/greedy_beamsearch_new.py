from .base import BaseBPE
import copy
import re
import tqdm
import itertools
import sys
import operator
from .base import BaseBPE

COLLAPSE_WHITESPACE = re.compile(r"\s+")

class Hypothesis():
    def __init__(self, merge_operations_parent, corpus_freqs_parent):
        self.merge_operations_parent = merge_operations_parent
        self.corpus_freqs_parent = corpus_freqs_parent
        self.merge_operations = None
        self.corpus_freqs = None
        self.pair_to_add = None
        self.possible_extensions = None

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
                # this should occur only once
                self.corpus_freqs = copy.deepcopy(self.corpus_freqs_parent)
            else:
                # this creates a new object so we're only borrowing parent's data
                self.corpus_freqs = BaseBPE.merge_vocab(self.pair_to_add, self.corpus_freqs_parent)

        return self.corpus_freqs

    def get_merge_operations(self):
        if self.merge_operations == None:
            self.merge_operations = copy.deepcopy(self.merge_operations_parent)
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

        # TODO: we don't need to store the history up until (exclusive) t-1
        beam_t = {0: [(0, Hypothesis(merge_operations, corpus_freqs))]}

        for t in tqdm.tqdm(range(1, vocab_size-len(merge_operations)+1)):
            beam = []
            for score, hyp in beam_t[t-1]:
                possible_pairs = hyp.get_possible_extensions()
                # TODO: to speed things up add only top-x
                # Should not matter much that there is lots of children because they are a light object with
                # lazy copying of their parents' data
                for pair in possible_pairs:
                    new_hyp = hyp.spawn_child(pair)
                    new_score = score + pair[1]
                    beam.append((new_score, new_hyp))

            beam.sort(key=operator.itemgetter(0), reverse=True)
            beam_t[t] = beam[:self.beam_n]
        
        beam_best = max(beam_t[t], key=operator.itemgetter(0))[1]
        
        self.merge_operations = beam_best.get_merge_operations()
        self.corpus_freqs = beam_best.get_corpus_freqs()

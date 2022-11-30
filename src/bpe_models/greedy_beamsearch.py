from .base import BaseBPE
import copy
import re
import tqdm
import itertools
import sys
import operator
from .base import BaseBPE

COLLAPSE_WHITESPACE = re.compile(r"\s+")

class BeamHypothesis():
    def __init__(self, merge_operations, corpus_freqs):
        self.merge_operations = merge_operations
        self.corpus_freqs = corpus_freqs
        self.score = 0
        self.extendable_pairs = list(BaseBPE.get_pairs(self.corpus_freqs).items())
        self.extendable_pairs.sort(key=lambda x: x[1], reverse=True)

    def add_pair(self, pair: tuple):
        pair_txt = pair[0]
        self.merge_operations.append(pair_txt[0] + " " + pair_txt[1])
        self.score += pair[1]        
        self.corpus_freqs = BaseBPE.merge_vocab(pair_txt, self.corpus_freqs)

        self.extendable_pairs = list(BaseBPE.get_pairs(self.corpus_freqs).items())
        self.extendable_pairs.sort(key=lambda x: x[1], reverse=True)

    def extend(self):
        # no pair left to extend
        if len(self.extendable_pairs) == 0:
            return False

        # best pair not yet extended from this beam
        best_pair = self.extendable_pairs.pop(0)

        # print("Choosing", best_pair)

        new_beam_hyp = self.spawn_child()
        new_beam_hyp.add_pair(best_pair)
        return new_beam_hyp

    def spawn_child(self):
        new_beam_hyp = copy.deepcopy(self)

        return new_beam_hyp

    def get_hash(self):
        return frozenset(self.merge_operations)

    def get_score(self):
        # use accumulated pair scores but could also directly use compressed corpus size
        return self.score

class GreedyBeamSearchBPE(BaseBPE):    
    def __init__(self, beam_n: int, beam_n_expand: int, **kwargs):
        self.beam_n = beam_n
        self.beam_n_expand = beam_n_expand

    def fit(self, corpus: str, vocab_size: int):
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

        queue= [(0, BeamHypothesis(merge_operations, corpus_freqs).spawn_child())]

        seen_vocab_hashes = set()
        # infinite iterator, provide a reasonable lower (upper?) bound estimate
        for i in tqdm.tqdm(itertools.count(), total=(vocab_size - len(merge_operations))):
            # beam_hyp_hash = beam_hyp.get_hash()

            queue_new = []
            for beam_score, beam_hyp in queue:
                for _ in range(self.beam_n_expand):
                    beam_hyp_new = beam_hyp.extend()
                    if beam_hyp_new != False:
                        beam_hyp_new_hash = beam_hyp_new.get_hash()
                        if beam_hyp_new_hash in seen_vocab_hashes:
                            pass
                        else:
                            queue_new.append((beam_hyp_new.get_score(), beam_hyp_new))
                            # print("Adding new child with len", len(beam_hyp_new.merge_operations), "of parent", len(beam_hyp.merge_operations))
                            seen_vocab_hashes.add(beam_hyp_new_hash)

            queue = queue_new
            # sort the queue
            queue.sort(key=operator.itemgetter(0), reverse=True)
            # crop the queue size
            queue = queue[:self.beam_n]

            if i % 100 == 0:
                print("Beam vocab sizes: ", [len(x[1].merge_operations) for x in queue])
                print("Beam vocab scores:", [x[0] for x in queue])
                sys.stdout.flush()

                # clean up hash set because nothing will have lower size
                min_len = min([len(x[1].merge_operations) for x in queue])
                seen_vocab_hashes = {v for v in seen_vocab_hashes if len(v) >= min_len}


            # stop if all beam hypotheses have enough operations to make it fair
            if all([
                len(beam_hyp.merge_operations) >= vocab_size
                for beam_score, beam_hyp in queue
            ]):
                break

        beam_hyp_best_score, beam_hyp_best = queue.pop(0)

        # choose the top
        self.merge_operations = beam_hyp_best.merge_operations
        self.corpus_freqs = beam_hyp_best.corpus_freqs

from .base import BaseBPE
import copy
import re
import tqdm
import itertools
import sys
import math

COLLAPSE_WHITESPACE = re.compile(r"\s+")

class Beam():
    def __init__(self, merge_operations, corpus_freqs):
        self.merge_operations = copy.deepcopy(merge_operations)
        self.corpus_freqs = copy.deepcopy(corpus_freqs)
        self.score = 0

    def add_pair(self, pair: tuple):
        pair_txt = pair[0]
        self.merge_operations.append(pair_txt[0] + " " + pair_txt[1])
        self.score += pair[1]

    def get_hash(self):
        return frozenset(self.merge_operations)

    def get_score(self):
        # use accumulated pair scores but could also directly use corpus size
        return self.score
        # use log to not be dominated by zipf but may not be important
        # return -math.log2(self.corpus.count(" "))
        # return -self.corpus.count(" ")

class GreedyBeamSearchOldBPE(BaseBPE):
    # TODO: the beam_n_expand is nonsense and should technically be 1
    # It should work out with that if we do initialization properly
    
    def __init__(self, beam_n: int, beam_n_expand: int, **kwargs):
        self.beam_n = beam_n
        self.beam_n_expand = beam_n_expand

    def choose_pair_to_merge(self, pairs):
        pairs = list(pairs.items())
        pairs.sort(key=lambda x: x[1], reverse=True)
        # TODO: there is no point in self.beam_n_expand > self.beam_n
        return pairs[:self.beam_n_expand]

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

        beams = [Beam(merge_operations, corpus_freqs)]

        # infinite iterator
        for i in tqdm.tqdm(itertools.count(), total=vocab_size - len(merge_operations)):
            # advance each beam
            beams_new = []
            beams_hash_duplicate = set()

            for beam in beams:
                pairs = self.get_pairs(beam.corpus_freqs)

                if not pairs:
                    break

                if i % 100 == 0:
                    print("Beam vocab size:", len(beam.merge_operations))
                    sys.stdout.flush()

                # choose max pair
                best_pairs = self.choose_pair_to_merge(pairs)

                # expand all pairs
                for pair in best_pairs:
                    beam_clone = copy.deepcopy(beam)
                    beam_clone.add_pair(pair)
                    # we can use beam.corpus_freq because we're not borrowing it mutably

                    # remove duplicates (very important because of order invariance)
                    beam_hash = beam_clone.get_hash()
                    if beam_hash in beams_hash_duplicate:
                        continue
                    else:
                        beams_hash_duplicate.add(beam_hash)
                        beam_clone.corpus_freqs = self.merge_vocab(pair[0], beam.corpus_freqs)
                        beams_new.append(beam_clone)


            # Rerank and cut off beams
            beams_new.sort(key=lambda beam: beam.get_score(), reverse=True)
            # print(f"Iteration {i} beam scores: ", ", ".join([f"{beam.get_score():.0f}" for beam in beams]))
            beams = beams_new[:self.beam_n]

            if len(beams[0].merge_operations) >= vocab_size:
                break

        beams_best = max(beams, key=lambda beam: beam.get_score())

        # choose the top
        self.merge_operations = beams_best.merge_operations
        self.corpus_freqs = beams_best.corpus_freqs

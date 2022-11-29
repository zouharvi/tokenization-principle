from .base import BaseBPE
import copy
import re
import tqdm
import itertools
import sys
import math

COLLAPSE_WHITESPACE = re.compile(r"\s+")

class Beam():
    def __init__(self, merge_operations, corpus, corpus_freqs):
        self.merge_operations = copy.deepcopy(merge_operations)
        self.corpus = copy.deepcopy(corpus)
        self.corpus_freqs = copy.deepcopy(corpus_freqs)

    def add_pair(self, pair: tuple):
        self.merge_operations.append(pair[0] + " " + pair[1])

        # TODO: this could potentially work also with corpus_freqs so we may use that to speed it up
        # However we need it for the hash
        self.corpus = self.corpus.replace(
            " " + pair[0] + " " + pair[1] + " ",
            " " + pair[0] + pair[1] + " ",
        )

    def score(self):
        # use log to not be dominated by zipf but may not be important
        return -math.log2(self.corpus.count(" "))
        # return -self.corpus.count(" ")

class GreedyBeamSearchBPE(BaseBPE):
    def __init__(self, beam_n: int, beam_n_expand: int, **kwargs):
        self.beam_n = beam_n
        self.beam_n_expand = beam_n_expand

    def choose_pair_to_merge(self, pairs):
        pairs = list(pairs.items())
        pairs.sort(key=lambda x: x[1], reverse=True)
        return [x[0] for x in pairs[:self.beam_n_expand]]

    @staticmethod
    def preprocess_beam_corpus(corpus):
        # join lines together
        corpus = " ".join(corpus)
        # collapse whitespace and newlines
        corpus = COLLAPSE_WHITESPACE.sub(" ", corpus)

        corpus = [" ".join(word) + " </w>" for word in corpus.split()]
        corpus = " " + " ".join(corpus) + " "
        return corpus

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

        corpus_preprocessed = self.preprocess_beam_corpus(corpus)

        beams = [Beam(merge_operations, corpus_preprocessed, corpus_freqs)]

        # infinite iterator
        for i in tqdm.tqdm(itertools.count(), total=vocab_size - len(merge_operations)):
            # advance each beam
            beams_new = []
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
                    beam_clone.corpus_freqs = self.merge_vocab(pair, beam.corpus_freqs)
                    beams_new.append(beam_clone)

            # remove duplicates (very important because of order invariance)
            beams_hash_duplicate = set()
            beams_new_original = []
            for beam in beams_new:
                # use processed corpora as a signature
                beam_hash = beam.corpus
                if beam_hash in beams_hash_duplicate:
                    continue
                else:
                    beams_hash_duplicate.add(beam_hash)
                    beams_new_original.append(beam)
            beams_new = beams_new_original

            # Rerank and cut off beams
            beams_new.sort(key=lambda beam: beam.score(), reverse=True)
            # print(f"Iteration {i} beam scores: ", ", ".join([f"{beam.score():.0f}" for beam in beams]))
            beams = beams_new[:self.beam_n]

            if len(beams[0].merge_operations) >= vocab_size:
                break

        # choose the top
        self.merge_operations = beams[0].merge_operations
        self.corpus_freqs = beams[0].corpus_freqs

import itertools
import re
from collections import Counter, defaultdict
import sys

import tqdm
COLLAPSE_WHITESPACE = re.compile(r"\s+")


class BaseBPE:
    def __init__(self, **kwargs):
        pass

    def build_vocab_freq(self, corpus: list[str]):
        """Build vocab from text corpus"""

        # join lines together
        corpus = " ".join(corpus)
        # collapse whitespace and newlines
        corpus = COLLAPSE_WHITESPACE.sub(" ", corpus)

        # separate each char in word by space and add mark end of token
        tokens = [" ".join(word) + " </w>" for word in corpus.split()]

        # count frequency of tokens in corpus
        vocab_freq = Counter(tokens)

        return vocab_freq

    def get_pairs(self, corpus_freqs: dict[str, int]) -> dict:
        """Get counts of pairs of consecutive symbols"""

        pairs = defaultdict(int)
        subword_vocab = ["UNK", "UNK</w>"]

        # paralelization here failed miserably
        for word, frequency in corpus_freqs.items():
            symbols = word.split(" ")
            subword_vocab += symbols

            # counting up occurrences of pairs
            for i in range(len(symbols) - 1):
                pairs[symbols[i], symbols[i + 1]] += frequency

        return pairs, set(subword_vocab)

    def merge_vocab(self, pair: tuple, corpus_freqs: dict) -> dict:
        """Merge all occurrences of a specific pair"""

        bigram_joined_space = re.escape(' '.join(pair))
        bigram_joined_space_re = re.compile(
            r'(?<!\S)' + bigram_joined_space + r'(?!\S)')

        # TODO: this "sanitization" will cause problems later and needs to be carefully checked
        bigram_joined = "".join(pair).replace("\\", "\\\\")

        corpus_freqs_new = {}
        # replace a pair in all vocabulary
        for word in corpus_freqs:
            w_out = bigram_joined_space_re.sub(
                bigram_joined, word).replace("\\\\", "\\")
            corpus_freqs_new[w_out] = corpus_freqs[word]

        return corpus_freqs_new

    def fit(self, corpus: str, vocab_size: int):
        corpus_freqs = self.build_vocab_freq(corpus)

        # infinite iterator
        for i in tqdm.tqdm(itertools.count(), total=vocab_size):
            # TODO: this is potentially heavy
            pairs, subword_vocab = self.get_pairs(corpus_freqs)

            if not pairs:
                break

            if len(subword_vocab) >= vocab_size:
                break

            if i % 100 == 0:
                print("Vocab size:", len(subword_vocab))
                sys.stdout.flush()

            # choose max pair
            best = self.choose_pair_to_merge(pairs)

            # TODO: this is potentially heavy
            corpus_freqs = self.merge_vocab(best, corpus_freqs)

        self.corpus_freqs = corpus_freqs
        self.subword_vocab = sorted(subword_vocab, key=lambda x: len(x))

    def encode_token(self, token):
        # ASSERT: subword_vocab is sorted from longest to shortest

        token_out = []
        while True:
            if len(token) == 0:
                break
            
            # TODO: this is a greedy decoding approach which is not optimal
            # The "greediness" in here is different from the BPE building
            found = False
            for subword in self.subword_vocab:
                if token.startswith(subword):
                    found = True
                    break
            
            if not found:
                subword = "UNK"

            token_out.append(subword)
            token = token[len(subword):]

        # no word ends with @@
        return "@@ ".join(token_out).removesuffix("</w>").strip().removesuffix("@@")

    def encode(self, corpus: list[str]) -> list[str]:
        # make sure it's sorted from longest to shortest
        self.subword_vocab = sorted(
            self.subword_vocab, key=lambda x: len(x),
            reverse=True
        )
        import multiprocess

        with multiprocess.Pool() as pool:
            out = pool.map(
                lambda line: " ".join([
                    self.encode_token(word + "</w>")
                    for word in line.split(" ")
                ]),
                tqdm.tqdm(corpus)
            )

        return out

    def save(self, path):
        with open(path, "w") as f:
            for word in self.subword_vocab:
                f.write(word + "\n")

    def load(self, path):
        with open(path, "r") as f:
            self.subword_vocab = [x.rstrip("\n") for x in f.readlines()]

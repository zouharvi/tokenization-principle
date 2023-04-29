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

    @staticmethod
    def get_pairs(corpus_freqs: dict[str, int]) -> dict:
        """Get counts of pairs of consecutive symbols"""

        pairs = defaultdict(int)

        # paralelization here failed miserably
        for word, frequency in corpus_freqs.items():
            symbols = word.split(" ")

            # counting up occurrences of pairs
            for i in range(len(symbols) - 1):
                pairs[symbols[i], symbols[i + 1]] += frequency

        return pairs

    @staticmethod
    def get_singles(corpus_freqs: dict[str, int]) -> dict:
        """Get counts of single symbols"""

        singles = defaultdict(int)

        # paralelization here failed miserably
        for word, frequency in corpus_freqs.items():
            symbols = word.split(" ")

            # counting up occurrences of pairs
            for i in range(len(symbols)):
                singles[symbols[i]] += frequency

        return singles

    @staticmethod
    def merge_vocab(pair: tuple, corpus_freqs: dict) -> dict:
        """Merge all occurrences of a specific pair"""

        bigram_joined_space = re.escape(' '.join(pair))
        bigram_joined_space_re = re.compile(
            r'(?<!\S)' + bigram_joined_space + r'(?!\S)'
        )

        # TODO: this "sanitization" will cause problems later and needs to be carefully checked
        bigram_joined = "".join(pair).replace("\\", "\\\\")

        corpus_freqs_new = {}
        # replace a pair in all vocabulary
        for word in corpus_freqs:
            w_out = bigram_joined_space_re.sub(
                bigram_joined, word
            ).replace("\\\\", "\\")
            corpus_freqs_new[w_out] = corpus_freqs[word]

        return corpus_freqs_new

    def fit(self, corpus: str, vocab_size: int):
        corpus_freqs = self.build_vocab_freq(corpus)

        # get initial characters
        pairs = self.get_pairs(corpus_freqs)
        # add all characters to subword vocab even if they are not merge operations
        self.merge_operations = list(
            {s for pair in pairs.keys() for s in pair})
        # add end characters
        self.merge_operations += list({
            pair[0] + pair[1]
            for pair in pairs.keys() if pair[1] == "</w>"
        })

        # infinite iterator
        for i in tqdm.tqdm(itertools.count(), total=vocab_size - len(self.merge_operations)):
            # TODO: this is potentially heavy
            pairs = self.get_pairs(corpus_freqs)
            singles = self.get_singles(corpus_freqs)

            # pairs_x = sorted(pairs.items(), key=lambda x: x[1], reverse=True)
            # if pairs_x[0][1] == pairs_x[1][1]:
            #     print()
            #     print("Indecision")
            #     print(pairs_x[0], pairs_x[1])
            #     print()

            if not pairs:
                break

            if i % 100 == 0:
                print("Vocab size:", len(self.merge_operations))
                sys.stdout.flush()

            # choose max pair
            best_pair = self.choose_pair_to_merge(pairs, singles)
            # TODO: this is potentially conflicting with antigreedy because that one can also remove subwords
            self.merge_operations.append(best_pair[0] + " " + best_pair[1])

            if len(self.merge_operations) >= vocab_size:
                break

            # TODO: this is potentially heavy
            corpus_freqs = self.merge_vocab(best_pair, corpus_freqs)

        self.corpus_freqs = corpus_freqs

    def encode_token_merge_operations(self, token: str, merge_operations: str, subword_vocab: set[str]):
        # ASSERT: merge_operations are sanitized

        # insert spaces between and around
        token = " " + " ".join(token) + "</w> "
        updated = True
        while updated:
            updated = False
            # break if word is done
            if " " not in token[1:-1]:
                break
            for merge_operation in merge_operations:
                if merge_operation in token:
                    # apply operation but make sure that it's still surrounded by spaces
                    token = token.replace(
                        merge_operation,
                        " " + merge_operation.replace(" ", "") + " "
                    )
                    token = " " + token.strip() + " "
                    updated = True
                    break

        # check that all individual components are in the vocabulary and do some post-processing
        token_new = "@@ ".join([
            x
            for x in token.strip(" ").split(" ")
        ]).removesuffix("</w>").removesuffix("@@")
        return token_new

    def encode_token_greedy_naive(self, token, subword_vocab: list[str]):
        # ASSERT: subword_vocab is sorted from longest to shortest

        token_out = []
        while True:
            if len(token) == 0:
                break

            # TODO: this is a greedy decoding approach which is not optimal
            # The "greediness" in here is different from the BPE building
            found = False
            for subword in subword_vocab:
                if token.startswith(subword):
                    found = True
                    break

            if not found:
                # subword = "UNK"
                # fall back to characters
                subword = list(token)
                token_out += subword
                break

            token_out.append(subword)
            token = token[len(subword):]

        # no word ends with @@
        return "@@ ".join(token_out).removesuffix("</w>").strip().removesuffix("@@")

    def preprocess_word(self, word):
        return word

    def encode_merge_operations(self, corpus: list[str]) -> list[str]:
        # make sure it's sorted from longest to shortest
        subword_vocab = set(
            self.merge_operations_to_subword_vocab(self.merge_operations)
        )
        # take only proper merge operations
        merge_operations = [
            " " + x + " " for x in self.merge_operations if " " in x]

        import multiprocess

        with multiprocess.Pool() as pool:
            out = pool.map(
                lambda line: " ".join([
                    self.encode_token_merge_operations(
                        self.preprocess_word(
                            word), merge_operations, subword_vocab
                    )
                    for word in line.split(" ")
                ]),
                tqdm.tqdm(corpus)
            )

        return out

    def encode_greedy_naive(self, corpus: list[str]) -> list[str]:
        # make sure it's sorted from longest to shortest
        subword_vocab = self.merge_operations_to_subword_vocab(
            self.merge_operations
        )
        subword_vocab.sort(key=lambda x: len(x), reverse=True)

        import multiprocess

        with multiprocess.Pool() as pool:
            out = pool.map(
                lambda line: " ".join([
                    self.encode_token_greedy_naive(
                        self.preprocess_word(word + "</w>"), subword_vocab)
                    for word in line.split(" ")
                ]),
                tqdm.tqdm(corpus)
            )

        return out

    @staticmethod
    def merge_operations_to_subword_vocab(merge_operations: list[str]) -> list[str]:
        subword_vocab = [
            x for pair in merge_operations
            for x in pair.split(" ")
        ]
        # add also combined words
        subword_vocab += [
            pair.replace(" ", "")
            for pair in merge_operations
            if " " in pair
        ]
        return subword_vocab

    def encode(self, corpus: list[str], method: str) -> list[str]:
        if method == "greedy_naive":
            return self.encode_greedy_naive(corpus)
        elif method == "merge_operations":
            return self.encode_merge_operations(corpus)
        else:
            raise Exception("Unknown decoding method " + method)

    def save(self, path):
        with open(path, "w") as f:
            for merge_operation in self.merge_operations:
                f.write(merge_operation + "\n")

    def load(self, path):
        with open(path, "r") as f:
            self.merge_operations = [x.rstrip("\n") for x in f.readlines()]

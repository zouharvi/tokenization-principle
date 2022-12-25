#!/usr/bin/env python3

import tqdm

class BaseLZW():
    def __init__(self, **kwargs):
        pass

    def fit(self, corpus, V):
        max_stored_length = 1
        # initialize with subwords of length 1
        dictionary = set("".join(corpus))
        # observed_dictionary = set("".join(corpus))

        break_loop = False
        observed_dictionary = set()
        while True:
            for line_i, line in enumerate(tqdm.tqdm(corpus)):
                if line == ["\n"]:
                    continue
                line_new = []
                j = 0
                while j < len(line):
                    for k in range(1, max_stored_length + 2)[::-1]:
                        if "".join(line[j:j + k]) in dictionary:
                            break

                    new_word = "".join(line[j:j + k])
                    observed_dictionary.add(new_word)
                    line_new.append(new_word)
                    # add continuation to the dictionary
                    dictionary.add("".join(line[j:j + k + 1]))
                    j += k

                corpus[line_i] = line_new

                if len(observed_dictionary) >= V:
                    break_loop = True
                    break

                # TODO: maybe should be in the inner-most loop
                max_stored_length = max([len(x) for x in dictionary])
            if break_loop:
                break

            print("Observing", len(observed_dictionary))

        print("Observing", len(observed_dictionary))
        # dictionary = set("".join(corpus))

        self.vocab = observed_dictionary
        return corpus


    def encode_token_greedy_naive(self, token, subword_vocab: list[str]):
        # ASSERT: subword_vocab is sorted from longest to shortest

        token_out = []
        while True:
            if len(token) == 0:
                break

            # TODO: this is a greedy decoding approach which is not optimal
            found = False
            for subword in subword_vocab:
                if token.startswith(subword):
                    found = True
                    break

            if not found:
                # subword = "UNK"
                subword = list(token)
                token_out += subword
                break

            token_out.append(subword)
            token = token[len(subword):]

        # no word ends with @@
        return "@@ ".join(token_out).strip().removesuffix("@@")


    def encode(self, corpus: list[str]) -> list[str]:
        # make sure it's sorted from longest to shortest
        subword_vocab = self.vocab
        subword_vocab.sort(key=lambda x: len(x), reverse=True)

        import multiprocess

        with multiprocess.Pool() as pool:
            out = pool.map(
                lambda line: " ".join([
                    self.encode_token_greedy_naive(
                        word, subword_vocab)
                    for word in line.split(" ")
                ]),
                tqdm.tqdm(corpus)
            )

        return out

    def load(self, path):
        with open(path, "r") as f:
            self.vocab = [x.rstrip("\n") for x in f.readlines()]
            self.vocab = [x for x in self.vocab if len(x) != 0]

    def save(self, path):
        with open(path, "w") as f:
            for word in self.vocab:
                f.write(word + "\n")


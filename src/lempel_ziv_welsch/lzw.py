#!/usr/bin/env python3


class LZW():
    def __init__(self):
        pass

    def fit(self, corpus, V):
        max_stored_length = 1
        # initialize with subwords of length 1
        dictionary = set(corpus)

        # measure observed dictionary count
        while len(set(corpus)) < V:
            j = 0
            corpus_new = []
            while j < len(corpus):                
                for k in range(1, max_stored_length+2)[::-1]:
                    if "".join(corpus[j:j+k]) in dictionary:
                        break
                
                corpus_new.append("".join(corpus[j:j+k]))
                dictionary.add("".join(corpus[j:j+k+1]))
                j += k
            corpus = corpus_new
            # TODO: maybe should be in the inside loop
            max_stored_length = max([len(x) for x in dictionary])

        dictionary = set(corpus)
        return corpus

model = LZW()

data = " ".join(open("data/CCrawl.de-en/train.tok.en").readlines()[:1000])
print(len(data))
data = model.fit(data, V=1024)
print(len(data))
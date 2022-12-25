from .base import BaseBPE, COLLAPSE_WHITESPACE, Counter

class GreedyCaptrickBPE(BaseBPE):
    def choose_pair_to_merge(self, pairs):
        return max(pairs, key=pairs.get)

    def build_vocab_freq(self, corpus: list[str]):
        """Build vocab from text corpus"""

        # join lines together
        corpus = " ".join(corpus)
        # collapse whitespace and newlines
        corpus = COLLAPSE_WHITESPACE.sub(" ", corpus)

        # separate each char in word by space and add mark end of token
        tokens = [" ".join(word) + " </w>" for word in corpus.split()]

        new_tokens = []
        for word in tokens:
            word = self.preprocess_word(word, extra_space=True)
            new_tokens.append(word)

        tokens = new_tokens

        # count frequency of tokens in corpus
        vocab_freq = Counter(tokens)

        return vocab_freq


    def preprocess_word(self, word, extra_space=False):
        if len(word) == 0:
            return word
            
        # all_lower = all(not x.isalpha() or x.islower() for x in word)
        all_upper = all(not x.isalpha() or x.isupper() for x in word)
        all_capital = word[0].isupper() and all(not x.isalpha() or x.islower() for x in word[1:])

        if extra_space:
            if all_upper:
                word = "$1 " + word.lower()
            elif all_capital:
                word = "$2 " + word.lower()
        else:
            if all_upper:
                word = "$1" + word.lower()
            elif all_capital:
                word = "$2" + word.lower()

        # print("preprocessing and returning", word)
        return word
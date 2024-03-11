import re
import collections
import tqdm
import argparse


def get_stats(vocab):
    pairs = collections.defaultdict(int)
    for word, freq in vocab.items():
        symbols = word.split()

        for i in range(len(symbols)-1):
            pairs[symbols[i], symbols[i+1]] += freq

    return pairs


def merge_vocab(pair, v_in):
    v_out = {}
    bigram = re.escape(' '.join(pair))
    p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
    for word in v_in:
        w_out = p.sub(''.join(pair), word)
        v_out[w_out] = v_in[word]
    return v_out


def bpe(vocab):
    num_merges = 2000

    # for _ in tqdm.tqdm(range(num_merges)):
    for _ in range(num_merges):
        pairs = get_stats(vocab)
        best = max(pairs, key=pairs.get)
        vocab = merge_vocab(best, vocab)


def build_vocab_freq(corpus: list[str]):
    """Build vocab from text corpus"""

    # join lines together
    corpus = " ".join(corpus)

    # separate each char in word by space and add mark end of token
    tokens = [" ".join(word) + " </w>" for word in corpus.split()]

    # count frequency of tokens in corpus
    vocab_freq = collections.Counter(tokens)

    return vocab_freq

args = argparse.ArgumentParser()
args.add_argument("-n", type=int, default=1000)
args = args.parse_args()

with open("data/CCrawl.de-en/train.tok.en", "r") as f:
    data = [x.rstrip("\n") for x in f.readlines()[:args.n]]
    vocab = build_vocab_freq(data)
    bpe(vocab)

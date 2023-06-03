#!/usr/bin/env python3

from dahuffman import HuffmanCodec
from statistics import covariance
import collections
import numpy as np

# load & split data
data_sent = [
    x.rstrip("\n").split()
    for x in open("data/dev.bpe.en", "r").readlines()
]
data_sent_train = data_sent[10000:]
data_sent_dev = data_sent[:10000]

# get bit encoder
data_word_freqs = collections.Counter(
    [x for line in data_sent_train for x in line]
)
codec = HuffmanCodec.from_frequencies(data_word_freqs)
# find word to bit len mapping
codec_lens = {
    k: l - 1
    for k, (l, code) in codec.get_code_table().items() if type(k) is str
}

data_sent_encoded_len = [
    # sum number of bits fo each word
    np.average([codec_lens[w] for w in x])
    for x in data_sent_train
    if x
]
# number of tokens in each sentence
data_sent_tok_count = [len(x) for x in data_sent_train if x]

print(covariance(data_sent_encoded_len, data_sent_tok_count))
# print(np.average(data_sent_encoded_len))
# print(np.average(data_sent_tok_count))

# sanity check
# print(data_sent_train[0])
# print([codec_lens[w] for w in data_sent_train[0]])
# print(np.average([codec_lens[w] for w in data_sent_train[0]]))
# print(len(data_sent_train[0]))
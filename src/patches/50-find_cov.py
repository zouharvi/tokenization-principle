#!/usr/bin/env python3

from dahuffman import HuffmanCodec
from statistics import covariance
import collections

# load & split data
data_sent = [
    x.rstrip("\n").split()
    for x in open("data/dev.bpe.en", "r").readlines()
]
data_sent_train = data_sent[:10000]
data_sent_dev = data_sent[10000:]

# get bit encoder
data_word_freqs = collections.Counter([x for line in data_sent_train for x in line])
codec = HuffmanCodec.from_frequencies(data_word_freqs)
# find word to bit len mapping
codec_lens = {
    k: l - 1
    for k, (l, code) in codec.get_code_table().items() if type(k) is str
}
max_len = 2+max(codec_lens.values())

data_sent_encoded_len = [
    # sum number of bits fo each word
    # replace unseen words with max + 2
    sum([codec_lens[w] if w in codec_lens else max_len for w in x])
    for x in data_sent_dev
]
# number of tokens in each sentence
data_sent_tok_count = [len(x) for x in data_sent_dev]

print(covariance(data_sent_encoded_len, data_sent_tok_count))

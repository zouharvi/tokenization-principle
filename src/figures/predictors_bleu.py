import numpy as np
import collections

def _predictor_subwords(data, vocab_size, extra_args):
    return data.count(" ") + data.count("\n")

def _predictor_seq_len(data, vocab_size, extra_args):
    return np.average([line.count(" ") + 1 for line in data.split("\n")])

def _predictor_bits(data, vocab_size, extra_args):
    return (data.count(" ") + data.count("\n")) * np.log2(vocab_size)

def _predictor_freq(data, vocab_size, extra_args):
    words_freqs = list(collections.Counter(data.split()).most_common())
    percentiles = np.arange(
        extra_args["freq_alpha_start"],
        # add epsilon to be included
        extra_args["freq_alpha_end"] + 0.001, step=0.05
    )
    freqs = np.average([
        words_freqs[
            min(int(len(words_freqs) * percentile), len(words_freqs) - 1)
        ][1]
        for percentile in percentiles
    ])
    return freqs

def _predictor_freq_prob(data, vocab_size, extra_args):
    words_freqs = list(collections.Counter(data.split()).most_common())
    total_subwords = sum([x[1] for x in words_freqs])
    percentiles = np.arange(
        extra_args["freq_alpha_start"],
        # add epsilon to be included
        extra_args["freq_alpha_end"] + 0.001, step=0.05
    )
    freqs = np.sum([
        words_freqs[
            min(int(len(words_freqs) * percentile), len(words_freqs) - 1)
        ][1]
        for percentile in percentiles
    ]) / total_subwords
    return freqs / np.log2(vocab_size)

def _predictor_renyi(data, vocab_size, extra_args):
    words_freqs = list(collections.Counter(data.split()).most_common())
    total_subwords = sum([x[1] for x in words_freqs])
    index_start = int(len(words_freqs) * extra_args["freq_alpha_start"])
    index_end = min(int(len(words_freqs) * extra_args["freq_alpha_end"]), len(words_freqs) - 1)
    
    freqs = np.log2(np.sum([
        (words_freqs[index][1] / total_subwords)**extra_args["renyi_alpha"]
        for index in range(index_start, index_end+1)
    ]))
    return freqs

# def _predictor_(data, vocab_size, extra_args):
# def _predictor_(data, vocab_size, extra_args):

PREDICTORS = {
    "subwords": (_predictor_subwords, "Subwords"),
    "seq_len": (_predictor_seq_len, "Sequence length"),
    "bits": (_predictor_bits, "Encoded bits"),
    "freq": (_predictor_freq, "TODO"),
    "freq_prob": (_predictor_freq_prob, "TODO"),
    "renyi": (_predictor_renyi, "TODO"),
}

def get_predictor(predictor):
    if predictor in PREDICTORS:
        return PREDICTORS[predictor]
    else:
        raise Exception(f"Unknown predictor {predictor}")
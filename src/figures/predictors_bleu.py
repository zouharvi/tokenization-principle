import numpy as np
import collections

def get_prob_distribution(data):
    words_freqs = list(collections.Counter(data.split()).most_common())
    total_subwords = sum([x[1] for x in words_freqs])
    freqs = [
        freq
        for word, freq in words_freqs
    ]
    probs = [
        freq/total_subwords
        for freq in freqs
    ]
    return freqs, probs

def _predictor_subwords(data, vocab_size, extra_args):
    return data.count(" ") + data.count("\n")

def _predictor_seq_len(data, vocab_size, extra_args):
    return np.average([line.count(" ") + 1 for line in data.split("\n")])

def _predictor_bits(data, vocab_size, extra_args):
    return (data.count(" ") + data.count("\n")) * np.log2(vocab_size)

def _predictor_freq(data, vocab_size, extra_args):
    words_freqs, probs = get_prob_distribution(data)

    start_i = min(
        int(len(words_freqs) * extra_args["freq_alpha_start"]),
        len(words_freqs) - 1
    )
    end_i = min(
        int(len(probs) * extra_args["freq_alpha_end"]), len(words_freqs) - 1
    )
    if start_i == end_i:
        start_i = max(0, start_i-1)
    if start_i == end_i:
        end_i = min(len(words_freqs)-1, end_i+1)

    indicies = range(start_i, end_i)

    # indicies = [int(x) for x in np.linspace(
    #     start_i, end_i + 0.001, 10
    # )]

    freqs = np.sum([
        words_freqs[i]
        for i in indicies
    ])

    return freqs

def _predictor_freq_prob(data, vocab_size, extra_args):
    words_freqs, probs = get_prob_distribution(data)

    start_i = min(
        int(len(probs) * extra_args["freq_alpha_start"]),
        len(probs) - 1
    )
    end_i = min(
        int(len(probs) * extra_args["freq_alpha_end"]), len(probs) - 1
    )
    if start_i == end_i:
        start_i = max(0, start_i-1)
    if start_i == end_i:
        end_i = min(len(probs)-1, end_i+1)

    indicies = range(start_i, end_i)

    indicies = [int(x) for x in np.linspace(
        start_i, end_i + 0.001, 10
    )]

    freqs = np.sum([
        probs[i]
        for i in indicies
    ])

    return freqs
    # return freqs/ np.log2(vocab_size)

def _predictor_renyi(data, vocab_size, extra_args):
    words_freqs, probs = get_prob_distribution(data)
    total_subwords = sum(words_freqs)
    index_start = int(len(words_freqs) * extra_args["freq_alpha_start"])
    index_end = min(int(len(words_freqs) * extra_args["freq_alpha_end"]), len(words_freqs) - 1)
    
    if extra_args["power"] == 1:
        scale = 1
    else:
        scale = 1/(1-extra_args["power"])

    freqs = np.log2(np.sum([
        (words_freqs[index] / total_subwords)**extra_args["power"]
        for index in range(index_start, index_end+1)
    ]))
    return scale*freqs

def _predictor_renyi_eff(data, vocab_size, extra_args):
    words_freqs, probs = get_prob_distribution(data)
    total_subwords = sum(words_freqs)
    index_start = int(len(words_freqs) * extra_args["freq_alpha_start"])
    index_end = min(int(len(words_freqs) * extra_args["freq_alpha_end"]), len(words_freqs) - 1)
    
    if extra_args["power"] == 1:
        scale = 1
    else:
        scale = 1/(1-extra_args["power"])

    freqs = scale*np.log2(np.sum([
        (words_freqs[index] / total_subwords)**extra_args["power"]
        for index in range(index_start, index_end+1)
    ]))/np.log2(vocab_size)
    return freqs

def _predictor_aggregate_deviation(data, vocab_size, extra_args):
    freqs, probs = get_prob_distribution(data)

    if extra_args["central_measure"] == "mean":
        central_measure = np.mean(freqs)
    elif extra_args["central_measure"] == "median":
        central_measure = np.median(freqs)
    elif extra_args["central_measure"] == "mode":
        _, counts = np.unique(freqs, return_counts=True)
        mode_index = np.argwhere(counts == np.max(counts))
        central_measure = freqs[mode_index]
    else:
        raise Exception("Unknown")
    
    # make sure the numbers are positive for non-integer powers
    if np.ceil(extra_args["power"]) != extra_args["power"]:
        vals = np.abs(np.array(freqs)-central_measure)**extra_args["power"]
    else:
        vals = (np.array(freqs)-central_measure)**extra_args["power"]

    if extra_args["aggregator"] == "mean":
        return np.average(vals)
    elif extra_args["aggregator"] == "median":
        return np.median(vals)
    elif extra_args["aggregator"] == "sqrt":
        # corresponds to standard deviation
        return np.sqrt(abs(np.sum(vals)/(len(vals)-1)))
    else:
        raise Exception("Unknown")

def _predictor_iqr(data, vocab_size, extra_args):
    freqs, probs = get_prob_distribution(data)

    if extra_args["base_vals"] == "freqs":
        vals = freqs
    elif extra_args["base_vals"] == "probs":
        vals = probs
    else:
        raise Exception("Unknown")
    
    index_start = int(len(freqs) * extra_args["freq_alpha_start"])
    index_end = min(int(len(freqs) * extra_args["freq_alpha_end"]), len(freqs) - 1)

    return vals[index_end] - vals[index_start]

def _predictor_entropy(data, vocab_size, extra_args):
    freqs, probs = get_prob_distribution(data)

    return -np.sum(probs * np.log2(probs))

def _predictor_entropy_eff(data, vocab_size, extra_args):
    freqs, probs = get_prob_distribution(data)

    return -np.sum(probs * np.log2(probs))/np.log2(vocab_size)

def _predictor_coefficient_variation(data, vocab_size, extra_args):
    freqs, probs = get_prob_distribution(data)
    
    return np.std(freqs)/np.average(freqs)

def _predictor_quartile_dispersion(data, vocab_size, extra_args):
    freqs, probs = get_prob_distribution(data)
    
    index_start = int(len(freqs) * extra_args["freq_alpha_start"])
    index_end = min(int(len(freqs) * extra_args["freq_alpha_end"]), len(freqs) - 1)

    # here we don't have to use base_vals because it's automatically normalized
    return (freqs[index_end] - freqs[index_start])/(freqs[index_end] + freqs[index_start])

# def _predictor_(data, vocab_size, extra_args):
# def _predictor_(data, vocab_size, extra_args):
# def _predictor_(data, vocab_size, extra_args):
# def _predictor_(data, vocab_size, extra_args):
# def _predictor_(data, vocab_size, extra_args):

PREDICTORS = {
    "subwords": (_predictor_subwords, "Subwords"),
    "seq_len": (_predictor_seq_len, "Sequence length"),
    "bits": (_predictor_bits, "Encoded bits with fixed-width"),
    "freq": (_predictor_freq, r"Frequency from 90$^{th}$ to 94$^{th}$ percentile"),
    "freq_prob": (_predictor_freq_prob, "TODO"),
    "renyi": (_predictor_renyi, r"Rényi entropy with $\alpha=3$"),
    "renyi_eff": (_predictor_renyi_eff, r"Rényi entropy efficiency with $\alpha=3$"),
    "agg_deviation": (_predictor_aggregate_deviation, "TODO"),
    "inter_quantile_range": (_predictor_iqr, "TODO"),
    "entropy": (_predictor_entropy, "Shannon entropy"),
    "entropy_eff": (_predictor_entropy_eff, "Shannon entropy efficiency"),
    "coefficient_variation": (_predictor_coefficient_variation, "TODO"),
    "quartile_dispersion": (_predictor_quartile_dispersion, "TODO"),
    # "": (_predictor_, "TODO"),
    # "": (_predictor_, "TODO"),
    # "": (_predictor_, "TODO"),
}

def get_predictor(predictor):
    if predictor in PREDICTORS:
        return PREDICTORS[predictor]
    else:
        raise Exception(f"Unknown predictor {predictor}")
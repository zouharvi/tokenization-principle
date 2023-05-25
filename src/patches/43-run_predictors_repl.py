#!/usr/bin/env python3

import collections
import json
import sys
sys.path.append("src")
from figures.predictors_bleu import get_predictor
from scipy.stats import pearsonr

data_metrics_1 = json.load(
    open("computed/metric_scores_repl.json", "r"),
)
data_metrics_2 = json.load(
    open("computed/metric_scores_repl_comet.json", "r"),
)

data_metrics = {
    lang: {
        signature: {
            **data_metrics_1[lang][signature],
            **data_metrics_2[lang][signature],
        }
        for signature in metric_scores_lang.keys()
    }
    for lang, metric_scores_lang in data_metrics_1.items()
}

PREDICTORS = [
    ("seq_len", {}),
    ("freq", {"freq_alpha_start": 0.8, "freq_alpha_end": 0.9}),
    ("entropy", {}),
    ("entropy_eff", {}),
    ("renyi", {"power": 3, "freq_alpha_start": 0, "freq_alpha_end": 1}),
    ("renyi_eff", {"power": 3, "freq_alpha_start": 0, "freq_alpha_end": 1}),
]

data_predictions = {
    "cs-en": collections.defaultdict(list),
    "de-en": collections.defaultdict(list),
}
for lang in ["de-en", "cs-en"]:
    for metric in ["bleu", "chrf", "bleurt", "comet"]:
        for signature, metric_scores in data_metrics[lang].items():
            data_predictions[lang][metric].append(metric_scores[metric])
    for predictor_key, predictor_args in PREDICTORS:
        predictor_fn, predictor_name = get_predictor(predictor_key)
        print(predictor_name, predictor_args)

        for signature, metric_scores in data_metrics[lang].items():
            signature_small = signature.replace("t", "").replace("v", "")
            vocab_size = float(signature_small.split("_")[1].replace("k", "000"))
            text_1 = open(f"data/model_bpe_random/{lang}/{signature_small}/dev.{lang.split('-')[0]}", "r").read()
            text_2 = open(f"data/model_bpe_random/{lang}/{signature_small}/dev.{lang.split('-')[1]}", "r").read()
            score = predictor_fn(text_1 + text_2, vocab_size, predictor_args)
            data_predictions[lang][predictor_key].append(score)

for predictor_key, predictor_args in PREDICTORS:
    predictor_fn, predictor_name = get_predictor(predictor_key)
    print(f"{predictor_name:<30}", end=" & ")
    for lang in ["de-en", "cs-en"]:
        for metric in ["bleu", "chrf", "bleurt", "comet"]:
            metric_list = data_predictions[lang][metric]
            pred_list = data_predictions[lang][predictor_key]
            score = pearsonr(metric_list, pred_list)[0]**2
            print(f"{score:.0%}".replace("%", "\\%"), end=" & ")
    print("\\\\")
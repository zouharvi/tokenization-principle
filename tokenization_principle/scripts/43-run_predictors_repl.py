#!/usr/bin/env python3

import collections
import json
from typing import Any

from scipy.stats import pearsonr
from tokenization_scorer import METRIC_TITLES
from tokenization_scorer import score as compute_score

with open("computed/metric_scores_repl.json", "r") as f:
    data_metrics_1 = json.load(f)
with open("computed/metric_scores_repl_comet.json", "r") as f:
    data_metrics_2 = json.load(f)

data_metrics = {
    lang: {
        signature: {
            **metric_scores_lang[signature],
            **data_metrics_2[lang][signature],
        }
        for signature in metric_scores_lang
    }
    for lang, metric_scores_lang in data_metrics_1.items()
}

PREDICTORS = [
    ("seq_len", {}),
    ("gowda", {"perc_start": 0.8, "perc_end": 0.9}),
    ("entropy", {}),
    ("entropy_eff", {}),
    ("renyi_entropy", {"power": 3}),
    ("renyi_eff", {"power": 3}),
]

data_predictions = {
    "cs-en": collections.defaultdict(list),
    "de-en": collections.defaultdict(list),
}
for lang in ["de-en", "cs-en"]:
    for metric in ["chrf", "bleurt", "comet"]:
        for signature, metric_scores in data_metrics[lang].items():
            data_predictions[lang][metric].append(metric_scores[metric])
    for predictor_key, predictor_args in PREDICTORS:
        predictor_name = METRIC_TITLES.get(predictor_key, predictor_key)
        print(predictor_name, predictor_args)

        for signature, metric_scores in data_metrics[lang].items():
            signature_small = signature.replace("t", "").replace("v", "")
            vocab_size = float(signature_small.split("_")[1].replace("k", "000"))
            with open(f"data/model_bpe_random/{lang}/{signature_small}/dev.{lang.split('-')[0]}", "r") as f:
                text_1 = f.read()
            with open(f"data/model_bpe_random/{lang}/{signature_small}/dev.{lang.split('-')[1]}", "r") as f:
                text_2 = f.read()
            val = compute_score(text_1 + text_2, metric=predictor_key, vocab=vocab_size, verbose=False, **predictor_args)
            data_predictions[lang][predictor_key].append(val)

for predictor_key, predictor_args in PREDICTORS:
    predictor_name = METRIC_TITLES.get(predictor_key, predictor_key)
    print(f"{predictor_name:<30}", end=" & ")
    for lang in ["de-en", "cs-en"]:
        for metric in ["chrf", "bleurt", "comet"]:
            metric_list = data_predictions[lang][metric]
            pred_list = data_predictions[lang][predictor_key]
            p_res: Any = pearsonr(metric_list, pred_list)
            score = float(p_res[0]) ** 2
            print(f"{score:.0%}".replace("%", "\\%"), end=" & ")
    print("\\\\")
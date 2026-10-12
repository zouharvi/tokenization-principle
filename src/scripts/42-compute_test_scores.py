#!/usr/bin/env python3

# sbatch --time=00-12 --ntasks=8 --mem-per-cpu=6G --gpus=1 \
#     --job-name="compute_test_scores" \
#     --output="logs_repl/compute_test_scores.log" \
#     --wrap="CUDA_VISIBLE_DEVICES=0 python3 ./src/scripts/42-compute_test_scores.py"
# sbatch --time=00-4 --ntasks=8 --mem-per-cpu=6G --gpus=1 \
#     --job-name="compute_test_scores_comet" \
#     --output="logs_repl/compute_test_scores_comet.log" \
#     --wrap="CUDA_VISIBLE_DEVICES=0 python3 ./src/scripts/42-compute_test_scores_comet.py"

import collections
import glob
import json

import evaluate
import numpy as np
import tqdm

metric_chrf = evaluate.load("chrf")
metric_bleu = evaluate.load("bleu")
metric_bleurt = evaluate.load("bleurt", module_type="metric")

metric_scores = {
    "cs-en": collections.defaultdict(list),
    "de-en": collections.defaultdict(list),
}

lines_src_all = {
    "cs-en": None,
    "de-en": None,
}

ref_files = {
    "cs-en": {
        "ref": "data/CCrawl.cs-en/dev.en",
        "src": "data/CCrawl.cs-en/dev.cs",
    },
    "de-en": {
        "ref": "data/CCrawl.de-en/dev.en",
        "src": "data/CCrawl.de-en/dev.de",
    },
}

for f in tqdm.tqdm(list(glob.glob("data_bin_ckpt_repl/*/model_bpe_random/*/dev_out.en"))):
    fs = f.split("/")
    lang = fs[1]
    if lang not in {"cs-en", "de-en"}:
        continue
    signature = fs[-2]
    print(lang, signature)
    with open(f, "r") as file_in:
        text_src = [
            l.rstrip().split("\t")[-1]
            for l in file_in if l.startswith("S")
        ]

    if lines_src_all[lang] is None:
        lines_src_all[lang] = set(text_src)
    else:
        lines_src_all[lang] &= set(text_src)

for f in tqdm.tqdm(list(glob.glob("data_bin_ckpt_repl/*/model_bpe_random/*/dev_out.en"))):
    fs = f.split("/")
    lang = fs[1]
    signature = fs[-2]
    signature_small = "_".join(signature.split("_")[1:])

    with open(f, "r") as file_in:
        lines_f = file_in.readlines()
    text_src = [
        l.rstrip().split("\t")[-1]
        for l in lines_f if l.startswith("S")
    ]
    text_hyp = [
        l.rstrip().split("\t")[-1]
        for l in lines_f if l.startswith("H")
    ]

    with open(ref_files[lang]["src"], "r") as file_src:
        gold_src = [l.rstrip() for l in file_src]
    with open(ref_files[lang]["ref"], "r") as file_ref:
        gold_ref = [l.rstrip() for l in file_ref]

    text = [
        (s, h)
        for s, h in zip(text_src, text_hyp) if s in lines_src_all[lang]
    ][:10000]
    text_gold = {s: r for s, r in zip(gold_src, gold_ref)}

    val_chrf = metric_chrf.compute(
        predictions=[h for s, h in text if s in text_gold],
        references=[[text_gold[s]] for s, h in text if s in text_gold]
    )["score"]
    print("chrf", val_chrf)
    val_bleu = metric_bleu.compute(
        predictions=[h for s, h in text if s in text_gold],
        references=[[text_gold[s]] for s, h in text if s in text_gold]
    )["bleu"]
    print("bleu", val_bleu)
    val_bleurt = np.average(metric_bleurt.compute(
        predictions=[h for s, h in text if s in text_gold],
        references=[text_gold[s] for s, h in text if s in text_gold]
    )["scores"])
    print("bleurt", val_bleurt)

    metric_scores[lang][signature_small].append({
        "bleu": val_bleu, "chrf": val_chrf, "bleurt": val_bleurt
    })

metrics = ["bleu", "chrf", "bleurt"]

metric_scores = {
    lang: {
        signature: {
            metric: np.average([x[metric] for x in signature_vals])
            for metric in metrics
        }
        for signature, signature_vals in metric_scores_lang.items()
    }
    for lang, metric_scores_lang in metric_scores.items()
}

with open("computed/metric_scores_repl.json", "w") as f_out:
    json.dump(
        metric_scores,
        f_out,
        ensure_ascii=False
    )

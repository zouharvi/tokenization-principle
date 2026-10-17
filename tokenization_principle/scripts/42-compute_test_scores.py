#!/usr/bin/env python3

# sbatch --time=00-12 --ntasks=8 --mem-per-cpu=6G --gpus=1 \
#     -J "compute_test_scores" \
#     --output=logs/%x.out --error=logs/%x.err \
#     --wrap="CUDA_VISIBLE_DEVICES=0 python3 ./src/scripts/42-compute_test_scores.py"
# sbatch --time=00-4 --ntasks=8 --mem-per-cpu=6G --gpus=1 \
#     -J "compute_test_scores_comet" \
#     --output=logs/%x.out --error=logs/%x.err \
#     --wrap="CUDA_VISIBLE_DEVICES=0 python3 ./src/scripts/42-compute_test_scores_comet.py"

import collections
import glob
import json

import evaluate
import numpy as np
import tqdm

metric_chrf = evaluate.load("chrf")

metric_scores = {
    "cs-en": collections.defaultdict(list),
    "de-en": collections.defaultdict(list),
}

lines_src_all: dict[str, set[str] | None] = {
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

dev_files = list(glob.glob("data/prepared/*/*/dev_out.en")) + list(glob.glob("data_bin_ckpt_repl/*/model_bpe_random/*/dev_out.en"))

def parse_lang_sig(path):
    fs = path.split("/")
    if path.startswith("data/prepared"):
        return fs[2], fs[3]
    return fs[1], fs[-2]

for f in tqdm.tqdm(dev_files):
    lang, signature = parse_lang_sig(f)
    if lang not in {"cs-en", "de-en"}:
        continue
    print(lang, signature)
    with open(f, "r") as file_in:
        lines_f = [l.rstrip() for l in file_in]
    if any(l.startswith("S\t") or l.startswith("S-") for l in lines_f):
        text_src = [
            l.split("\t")[-1]
            for l in lines_f if l.startswith("S")
        ]
    else:
        with open(ref_files[lang]["src"], "r") as file_src:
            text_src = [l.rstrip() for l in file_src if 1 <= len(l.rstrip()) <= 100][:len(lines_f)]

    if lines_src_all[lang] is None:
        lines_src_all[lang] = set(text_src)
    else:
        lines_src_all[lang] &= set(text_src) # type: ignore

for f in tqdm.tqdm(dev_files):
    lang, signature = parse_lang_sig(f)
    if lang not in {"cs-en", "de-en"}:
        continue
    signature_small = "_".join(signature.split("_")[1:])

    with open(f, "r") as file_in:
        lines_f = [l.rstrip() for l in file_in]
    if any(l.startswith("H\t") or l.startswith("H-") for l in lines_f):
        text_src = [
            l.split("\t")[-1]
            for l in lines_f if l.startswith("S")
        ]
        text_hyp = [
            l.split("\t")[-1]
            for l in lines_f if l.startswith("H")
        ]
    else:
        text_hyp = lines_f
        with open(ref_files[lang]["src"], "r") as file_src:
            text_src = [l.rstrip() for l in file_src if 1 <= len(l.rstrip()) <= 100][:len(text_hyp)]

    with open(ref_files[lang]["src"], "r") as file_src:
        gold_src = [l.rstrip() for l in file_src]
    with open(ref_files[lang]["ref"], "r") as file_ref:
        gold_ref = [l.rstrip() for l in file_ref]

    curr_lines_all = lines_src_all[lang] or set()
    text = [
        (s, h)
        for s, h in zip(text_src, text_hyp) if s in curr_lines_all
    ][:10000]
    text_gold = {s: r for s, r in zip(gold_src, gold_ref)}

    res_chrf = metric_chrf.compute(
        predictions=[h for s, h in text if s in text_gold],
        references=[[text_gold[s]] for s, h in text if s in text_gold]
    )
    val_chrf = res_chrf["score"] if res_chrf else 0.0
    print("chrf", val_chrf)
    metric_scores[lang][signature_small].append({
        "chrf": val_chrf
    })

metrics = ["chrf"]

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

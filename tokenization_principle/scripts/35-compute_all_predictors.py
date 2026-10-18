#!/usr/bin/env python3

import argparse
import glob
import json
import os
import tomllib
from pathlib import Path

from tokenization_scorer import score

METRICS_TO_COMPUTE = [
    ("renyi_eff", {"power": 2.5}),
    ("renyi_eff_p3", {"power": 3.0}),
    ("renyi_entropy", {"power": 2.5}),
    ("shannon_eff", {}),
    ("shannon_entropy", {}),
    ("seq_len", {}),
    ("subwords", {}),
    ("bits", {}),
    ("gowda", {"perc_start": 0.03, "perc_end": 0.83}),
    ("coefficient_variation", {}),
    ("quartile_dispersion", {"perc_start": 0.25, "perc_end": 0.75}),
    ("inter_quantile_range", {"perc_start": 0.25, "perc_end": 0.75}),
]


def dump_toml(data_dict, file_path):
    os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
    with open(file_path, "w") as f:
        f.write("# Tokenization Intrinsics and Downstream MT Evaluations\n\n")
        for lang_pair, configs in sorted(data_dict.items()):
            for config_name, metrics in sorted(configs.items()):
                f.write(f'["{lang_pair}"."{config_name}"]\n')
                for k, v in sorted(metrics.items()):
                    if isinstance(v, float):
                        f.write(f"{k} = {v:.6f}\n")
                    elif isinstance(v, int):
                        f.write(f"{k} = {v}\n")
                    elif isinstance(v, str):
                        f.write(f'{k} = "{v}"\n')
                    else:
                        f.write(f"{k} = {v}\n")
                f.write("\n")


def parse_vocab_size(config_name):
    # Extracts vocab size from name like unigram_v4000 or random_bpe_t0.4_v8000
    for part in config_name.split("_"):
        if part.startswith("v") and part[1:].isdigit():
            return int(part[1:])
    return None


def main():
    parser = argparse.ArgumentParser(description="Compute all tokenization predictors and save in single TOML file.")
    parser.add_argument("--prepared-dir", default="data/prepared", help="Directory containing prepared tokenized splits")
    parser.add_argument("--output-toml", default="computed/tokenization_evals.toml", help="Path to output TOML dictionary")
    parser.add_argument("--evals-log", default="computed/apply_bpe_all_chrf.jsonl", help="Optional log with MT evaluation scores")
    parser.add_argument("--split", default="dev", choices=["dev", "test", "train"], help="Which dataset split to evaluate intrinsics on")
    args = parser.parse_args()

    # Load existing TOML if present
    data_dict = {}
    if os.path.isfile(args.output_toml):
        with open(args.output_toml, "rb") as f:
            data_dict = tomllib.load(f)

    # Optional: load MT performance scores if log exists
    mt_perf = {}
    if os.path.isfile(args.evals_log):
        with open(args.evals_log, "r") as f:
            for line in f:
                item = json.loads(line)
                model = item.get("model")
                chrf = item.get("best_chrf", item.get("perf"))
                if model and chrf is not None:
                    mt_perf[model] = float(chrf)

    # Traverse prepared tokenized datasets
    pattern = os.path.join(args.prepared_dir, "*", "*")
    found_dirs = sorted(glob.glob(pattern))

    for config_dir in found_dirs:
        if not os.path.isdir(config_dir):
            continue
        p = Path(config_dir)
        config_name = p.name
        lang_pair = p.parent.name

        if "-" not in lang_pair:
            continue

        l1, l2 = lang_pair.split("-")
        f1 = os.path.join(config_dir, f"{args.split}.{l1}")
        f2 = os.path.join(config_dir, f"{args.split}.{l2}")

        if not (os.path.isfile(f1) and os.path.isfile(f2)):
            continue

        print(f"Evaluating {lang_pair}/{config_name} ({args.split})...")
        with open(f1, "r", encoding="utf-8") as f:
            text1 = f.read()
        with open(f2, "r", encoding="utf-8") as f:
            text2 = f.read()

        combined_text = text1 + "\n" + text2
        vocab_size = parse_vocab_size(config_name)

        if lang_pair not in data_dict:
            data_dict[lang_pair] = {}
        if config_name not in data_dict[lang_pair]:
            data_dict[lang_pair][config_name] = {}

        entry = data_dict[lang_pair][config_name]
        if vocab_size is not None:
            entry["vocab_size"] = vocab_size

        for metric_name, metric_kwargs in METRICS_TO_COMPUTE:
            kwargs = dict(metric_kwargs)
            if vocab_size is not None:
                kwargs["vocab"] = vocab_size
            kwargs["verbose"] = False
            base_metric = "renyi_eff" if metric_name.startswith("renyi_eff") else metric_name
            val = score(combined_text, metric=base_metric, **kwargs)
            entry[metric_name] = float(val)

        # Merge MT performance if available
        sig_lookup = f"{config_name}.{lang_pair}"
        if sig_lookup in mt_perf:
            entry["mt_chrf"] = mt_perf[sig_lookup]
        elif config_name in mt_perf:
            entry["mt_chrf"] = mt_perf[config_name]

    dump_toml(data_dict, args.output_toml)
    print(f"Saved tokenization evaluations and intrinsics to {args.output_toml}")


if __name__ == "__main__":
    main()

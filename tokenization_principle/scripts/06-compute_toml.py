#!/usr/bin/env python3

import argparse
from pathlib import Path

import sacrebleu
import torch
from comet import download_model, load_from_checkpoint
from tokenization_scorer import score

METRICS = [
    ("renyi_eff", "renyi_efficiency", {"power": 2.5}),
    ("renyi_eff_p3", "renyi_efficiency", {"power": 3.0}),
    ("renyi_entropy", "renyi_entropy", {"power": 2.5}),
    ("shannon_eff", "shannon_efficiency", {}),
    ("shannon_entropy", "shannon_entropy", {}),
    ("seq_len", "seq_len", {}),
    ("subwords", "subwords", {}),
    ("bits", "bits", {}),
    ("gowda", "gowda", {"perc_start": 0.03, "perc_end": 0.83}),
    ("coefficient_variation", "coefficient_variation", {}),
    ("quartile_dispersion", "quartile_dispersion", {"perc_start": 0.25, "perc_end": 0.75}),
    ("inter_quantile_range", "inter_quantile_range", {"perc_start": 0.25, "perc_end": 0.75}),
]


def detok(path: Path) -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip().replace("@@ ", "") for line in f]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", required=True, type=Path, help="Prepared data directory (e.g. data/prepared/de-en/lzw_v16000)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output.toml if it already exists")
    args = parser.parse_args()

    out_file = args.dir / "output.toml"
    if out_file.exists() and not args.overwrite:
        print(f"{out_file} already exists. Use --overwrite to recompute.")
        return

    l1, l2 = args.dir.parent.name.split("-")
    lines_out: list[str] = []

    # 1. Intrinsic tokenization metrics on each side of train
    for lang in (l1, l2):
        train_text = (args.dir / f"train.{lang}").read_text(encoding="utf-8")
        lines_out.append(f"[intrinsics.{lang}]")
        for name, base, kwargs in METRICS:
            val = float(score(train_text, metric=base, **kwargs))
            lines_out.append(f"{name} = {val:.6f}")
        lines_out.append("")

    # 2. Downstream MT quality across seeds
    hyp_files = sorted((args.dir / "checkpoints").glob("test_s*.txt"))
    if hyp_files:
        srcs = detok(args.dir / f"test.{l1}")
        refs = detok(args.dir / f"test.{l2}")
        comet_model = load_from_checkpoint(download_model("Unbabel/wmt22-comet-da"))
        gpus = 1 if torch.cuda.is_available() else 0

        for hyp_path in hyp_files:
            seed_key = hyp_path.stem.removeprefix("test_")
            hyps = detok(hyp_path)
            chrf = sacrebleu.corpus_chrf(hyps, [refs]).score
            bleu = sacrebleu.corpus_bleu(hyps, [refs]).score
            comet_in = [{"src": s, "mt": h, "ref": r} for s, h, r in zip(srcs, hyps, refs)]
            comet = float(comet_model.predict(comet_in, batch_size=64, gpus=gpus)["system_score"])

            lines_out.append(f"[mt.{seed_key}]")
            lines_out.append(f"chrf = {chrf:.4f}")
            lines_out.append(f"bleu = {bleu:.4f}")
            lines_out.append(f"comet = {comet:.6f}")
            lines_out.append("")

    out_file.write_text("\n".join(lines_out), encoding="utf-8")
    print(f"Wrote {out_file}")


if __name__ == "__main__":
    main()

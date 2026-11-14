#!/usr/bin/env python3

import argparse
from pathlib import Path
from typing import Any

import sacrebleu
import toml
import torch
from comet import download_model, load_from_checkpoint
from sacremoses import MosesDetokenizer
from tokenization_scorer import score

RENYI_POWERS = [0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]

METRICS = (
    [(f"renyi_eff_p{p}", "renyi_efficiency", {"power": p}) for p in RENYI_POWERS]
    + [(f"renyi_entropy_p{p}", "renyi_entropy", {"power": p}) for p in RENYI_POWERS]
    + [
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
)


def detok(path: Path, lang: str) -> list[str]:
    md = MosesDetokenizer(lang=lang)
    with open(path, "r", encoding="utf-8") as f:
        return [md.detokenize(line.strip().replace("@@ ", "").split()) for line in f]


def eval_intrinsics(text: str) -> dict[str, Any]:
    res: dict[str, Any] = {"vocab_size_observed": len(set(text.split()))}
    for name, base, kwargs in METRICS:
        res[name] = float(score(text, metric=base, **kwargs)) # type: ignore
    return res


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", required=True, type=Path, help="Prepared data directory (e.g. data/prepared/de-en/lzw_v16000)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output.toml if it already exists")
    args = parser.parse_args()

    out_file = args.dir / "output.toml"
    if out_file.exists() and not args.overwrite:
        print(f"{out_file} already exists. Use --overwrite to recompute.")
        return

    method, vocab_str = args.dir.name.rsplit("_v", 1)
    l1, l2 = args.dir.parent.name.split("-")
    text1 = (args.dir / f"train.{l1}").read_text(encoding="utf-8")
    text2 = (args.dir / f"train.{l2}").read_text(encoding="utf-8")

    data: dict[str, Any] = {
        "tokenizer": {
            "method": method,
            "vocab_size": int(vocab_str),
        },
        "intrinsics": {
            l1: eval_intrinsics(text1),
            l2: eval_intrinsics(text2),
            "joint": eval_intrinsics(text1 + "\n" + text2),
        },
    }

    ckpt_dir = args.dir / "checkpoints"
    hyp_files = sorted(ckpt_dir.glob("test_s*.txt"))
    if hyp_files:
        data["mt"] = {}
        srcs = detok(args.dir / f"test.{l1}", l1)
        refs = detok(args.dir / f"test.{l2}", l2)
        comet_model = load_from_checkpoint(download_model("Unbabel/wmt22-comet-da"))
        gpus = 1 if torch.cuda.is_available() else 0

        for hyp_path in hyp_files:
            seed_key = hyp_path.stem.removeprefix("test_")
            raw_tokens = hyp_path.read_text(encoding="utf-8").split()
            unk_rate = raw_tokens.count("<unk>") / max(1, len(raw_tokens))

            ckpt_path = ckpt_dir / f"checkpoint_best_{seed_key}.pt"
            best_epoch = int(torch.load(ckpt_path, map_location="cpu")["epoch"]) if ckpt_path.exists() else -1

            hyps = detok(hyp_path, l2)
            chrf = float(sacrebleu.corpus_chrf(hyps, [refs]).score)
            bleu = float(sacrebleu.corpus_bleu(hyps, [refs]).score)
            comet_in = [{"src": s, "mt": h, "ref": r} for s, h, r in zip(srcs, hyps, refs)]
            comet = float(comet_model.predict(comet_in, batch_size=64, gpus=gpus)["system_score"])

            data["mt"][seed_key] = {
                "best_epoch": best_epoch,
                "unk_rate": unk_rate,
                "chrf": chrf,
                "bleu": bleu,
                "comet": comet,
            }

    with open(out_file, "w", encoding="utf-8") as f:
        toml.dump(data, f)
    print(f"Wrote {out_file}")


if __name__ == "__main__":
    main()

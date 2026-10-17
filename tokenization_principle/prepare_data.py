#!/usr/bin/env python3
"""
Unified Tokenizer Preparation Pipeline.
Fits and applies any supported tokenizer (BPE, LZW, Morfessor, Unigram, WordPiece)
directly on train, dev, and test splits into a clean, standardized folder structure.
"""

import argparse
import os
import random
import subprocess
import tempfile
import time

from tokenization_principle.lempel_ziv_welsch.lzw import BaseLZW
from tokenization_principle.utils import get_model


def parse_args():
    parser = argparse.ArgumentParser(description="Unified Tokenizer Preparation Pipeline")
    parser.add_argument(
        "--tokenizer",
        required=True,
        choices=[
            "unigram",
            "wordpiece",
            "hf_bpe",
            "greedy_bpe",
            "antigreedy_bpe",
            "random_bpe",
            "lzw",
            "morfessor",
        ],
        help="Tokenizer method to fit and apply",
    )
    parser.add_argument("--vocab-size", type=int, default=8000, help="Target vocabulary size")
    parser.add_argument("--lang-pair", required=True, help="Language pair, e.g. de-en, cs-en, zh-en")
    parser.add_argument("--input-dir", required=True, help="Directory containing pretokenized data (.tok files)")
    parser.add_argument("--output-dir", required=True, help="Directory to save tokenized splits and model")

    # Line limits
    parser.add_argument("--fit-lines", type=int, default=100000, help="Number of training lines to fit tokenizer on")
    parser.add_argument("--train-lines", type=int, default=10000000, help="Max lines to tokenize for train split")
    parser.add_argument("--dev-lines", type=int, default=50000, help="Max lines to tokenize for dev split")
    parser.add_argument("--test-lines", type=int, default=50000, help="Max lines to tokenize for test split")

    # Hyperparameters
    parser.add_argument("--temperature", type=float, default=1.0, help="Randomness temperature for random_bpe")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Hugging Face Tokenizers (Unigram, WordPiece, BPE)
# ---------------------------------------------------------------------------
def fit_and_apply_hf(tokenizer_type, vocab_size, train_fit_lines, splits_to_process, out_dir):
    import tokenizers
    import tokenizers.models
    import tokenizers.pre_tokenizers
    import tokenizers.trainers

    if tokenizer_type == "unigram":
        tok_model = tokenizers.models.Unigram()
        trainer = tokenizers.trainers.UnigramTrainer(
            vocab_size=vocab_size, special_tokens=["[UNK]"], unk_token="[UNK]"
        )
    elif tokenizer_type == "wordpiece":
        tok_model = tokenizers.models.WordPiece()
        tok_model.unk_token = "[UNK]"
        trainer = tokenizers.trainers.WordPieceTrainer(
            vocab_size=vocab_size, special_tokens=["[UNK]"]
        )
    elif tokenizer_type == "hf_bpe":
        tok_model = tokenizers.models.BPE()
        tok_model.unk_token = "[UNK]"
        trainer = tokenizers.trainers.BpeTrainer(
            vocab_size=vocab_size, special_tokens=["[UNK]"]
        )
    else:
        raise ValueError(f"Unknown HF tokenizer: {tokenizer_type}")

    tokenizer = tokenizers.Tokenizer(tok_model)
    tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.WhitespaceSplit()

    # Write fit corpus to temporary file
    with tempfile.NamedTemporaryFile("w+", encoding="utf-8") as tmp:
        for line in train_fit_lines:
            tmp.write(line + "\n")
        tmp.flush()
        print(f"Training HF {tokenizer_type} on {len(train_fit_lines)} lines...")
        tokenizer.train([tmp.name], trainer)

    model_path = os.path.join(out_dir, "model.json")
    tokenizer.save(model_path)

    # Encode splits
    for in_path, out_path, max_n in splits_to_process:
        print(f"Applying {tokenizer_type} to {in_path} -> {out_path} ({max_n} lines max)...")
        with open(in_path, "r", encoding="utf-8") as f_in:
            lines = [l.rstrip("\r\n") for l in f_in.readlines()[:max_n]]

        encodings = tokenizer.encode_batch(lines)
        with open(out_path, "w", encoding="utf-8") as f_out:
            for enc in encodings:
                tokens = []
                last_right = None
                for token, offset in zip(enc.tokens, enc.offsets):
                    token = token.removeprefix("##")
                    if last_right != offset[0]:
                        tokens.append(token)
                    else:
                        tokens.append("@@" + token)
                    last_right = offset[1]
                formatted = " ".join(tokens).replace(" @@", "@@ ")
                f_out.write(formatted + "\n")


# ---------------------------------------------------------------------------
# Custom BPE (Greedy, Antigreedy, Random)
# ---------------------------------------------------------------------------
def fit_and_apply_bpe(bpe_kind, vocab_size, train_fit_lines, splits_to_process, out_dir, temp=1.0, seed=0):
    model_name = "random" if bpe_kind == "random_bpe" else ("antigreedy" if bpe_kind == "antigreedy_bpe" else "greedy")
    dist = "softmax" if bpe_kind == "random_bpe" else "uniform"

    print(f"Fitting BPE ({model_name}, temp={temp}, seed={seed}, V={vocab_size}) on {len(train_fit_lines)} lines...")
    model_cls = get_model(model_name)
    model = model_cls(
        seed=seed,
        randomness_dist=dist,
        randomness_temp=temp,
    )
    # fit expects list of strings with newlines
    model.fit([l + "\n" for l in train_fit_lines], vocab_size=vocab_size)

    vocab_file = os.path.join(out_dir, "model.bpe_merges")
    model.save(vocab_file)

    for in_path, out_path, max_n in splits_to_process:
        print(f"Applying BPE to {in_path} -> {out_path} ({max_n} lines max)...")
        with open(in_path, "r", encoding="utf-8") as f_in:
            lines = [l.rstrip("\r\n") for l in f_in.readlines()[:max_n]]

        encoded = model.encode(lines, method="greedy_naive")
        with open(out_path, "w", encoding="utf-8") as f_out:
            f_out.writelines(l + "\n" for l in encoded)


# ---------------------------------------------------------------------------
# LZW
# ---------------------------------------------------------------------------
def fit_and_apply_lzw(vocab_size, train_fit_lines, splits_to_process, out_dir):
    print(f"Fitting LZW (V={vocab_size}) on {len(train_fit_lines)} lines...")
    model = BaseLZW()

    # LZW fit format expects list of token lists
    data = [[w for w in line.split(" ")] + ["\n"] for line in train_fit_lines]
    data_flat = [w for line in data for w in line]
    model.fit(data_flat, V=vocab_size)

    vocab_file = os.path.join(out_dir, "model.vocab")
    model.save(vocab_file)

    for in_path, out_path, max_n in splits_to_process:
        print(f"Applying LZW to {in_path} -> {out_path} ({max_n} lines max)...")
        with open(in_path, "r", encoding="utf-8") as f_in:
            lines = [l.rstrip("\r\n") for l in f_in.readlines()[:max_n]]

        encoded = model.encode(lines)
        with open(out_path, "w", encoding="utf-8") as f_out:
            f_out.writelines(l + "\n" for l in encoded)


# ---------------------------------------------------------------------------
# Morfessor
# ---------------------------------------------------------------------------
def fit_and_apply_morfessor(vocab_size, train_fit_lines, splits_to_process, out_dir):
    model_path = os.path.join(out_dir, "model.pkl")

    with tempfile.NamedTemporaryFile("w+", encoding="utf-8") as tmp_fit:
        for l in train_fit_lines:
            tmp_fit.write(l + "\n")
        tmp_fit.flush()

        cmd_train = f"morfessor-train {tmp_fit.name} -s {model_path} --num-morph-types {vocab_size}"
        print(f"Fitting Morfessor: {cmd_train}")
        subprocess.run(cmd_train, shell=True, check=True)

    for in_path, out_path, max_n in splits_to_process:
        print(f"Applying Morfessor to {in_path} -> {out_path} ({max_n} lines max)...")
        with tempfile.NamedTemporaryFile("w+", encoding="utf-8") as tmp_in, \
             tempfile.NamedTemporaryFile("r+", encoding="utf-8") as tmp_out:
            with open(in_path, "r", encoding="utf-8") as f_in:
                lines = [l.rstrip("\r\n") for l in f_in.readlines()[:max_n]]
            for l in lines:
                tmp_in.write(l + "\n")
            tmp_in.flush()

            cmd_seg = f"morfessor-segment {tmp_in.name} --load {model_path} --output {tmp_out.name}"
            subprocess.run(cmd_seg, shell=True, check=True)

            tmp_out.seek(0)
            segmented_words = [l.rstrip("\r\n") for l in tmp_out.readlines()]

            # Reconstruct sentences formatted with @@
            with open(out_path, "w", encoding="utf-8") as f_out:
                word_idx = 0
                for orig_line in lines:
                    words = orig_line.split()
                    line_subwords = []
                    for _ in range(len(words)):
                        if word_idx < len(segmented_words):
                            morphs = segmented_words[word_idx].split()
                            if len(morphs) > 1:
                                line_subwords.append("@@ ".join(morphs))
                            elif morphs:
                                line_subwords.append(morphs[0])
                            word_idx += 1
                    f_out.write(" ".join(line_subwords) + "\n")


# ---------------------------------------------------------------------------
# Main Pipeline Entry Point
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    langs = args.lang_pair.split("-")
    if len(langs) != 2:
        raise ValueError(f"Expected lang-pair in format L1-L2, got {args.lang_pair}")
    l1, l2 = langs[0], langs[1]

    # Find pretokenized input files
    def get_input(split, lang):
        p = os.path.join(args.input_dir, f"{split}.tok.{lang}")
        if not os.path.exists(p):
            p_fallback = os.path.join(args.input_dir, f"{split}.{lang}")
            if os.path.exists(p_fallback):
                return p_fallback
        return p

    train_l1 = get_input("train", l1)
    train_l2 = get_input("train", l2)
    dev_l1 = get_input("dev", l1)
    dev_l2 = get_input("dev", l2)
    test_l1 = get_input("test", l1)
    test_l2 = get_input("test", l2)

    if not os.path.exists(train_l1) or not os.path.exists(train_l2):
        raise FileNotFoundError(f"Training input files missing: {train_l1} or {train_l2}")

    # Read training lines for fitting (interleaved across both languages)
    print(f"Reading {args.fit_lines} lines each from {train_l1} and {train_l2}...")
    with open(train_l1, "r", encoding="utf-8") as f1, open(train_l2, "r", encoding="utf-8") as f2:
        lines_l1 = [l.strip() for l in f1.readlines()[:args.fit_lines]]
        lines_l2 = [l.strip() for l in f2.readlines()[:args.fit_lines]]

    # Interleave to build a balanced joined vocabulary
    train_fit_lines = [item for pair in zip(lines_l1, lines_l2) for item in pair]

    # Define splits to process: (in_file, out_file, max_lines)
    splits = [
        (train_l1, os.path.join(args.output_dir, f"train.{l1}"), args.train_lines),
        (train_l2, os.path.join(args.output_dir, f"train.{l2}"), args.train_lines),
        (dev_l1, os.path.join(args.output_dir, f"dev.{l1}"), args.dev_lines),
        (dev_l2, os.path.join(args.output_dir, f"dev.{l2}"), args.dev_lines),
    ]

    if os.path.exists(test_l1) and os.path.exists(test_l2):
        splits.append((test_l1, os.path.join(args.output_dir, f"test.{l1}"), args.test_lines))
        splits.append((test_l2, os.path.join(args.output_dir, f"test.{l2}"), args.test_lines))

    start_time = time.time()
    t = args.tokenizer
    if t in {"unigram", "wordpiece", "hf_bpe"}:
        fit_and_apply_hf(t, args.vocab_size, train_fit_lines, splits, args.output_dir)
    elif t in {"greedy_bpe", "antigreedy_bpe", "random_bpe"}:
        fit_and_apply_bpe(t, args.vocab_size, train_fit_lines, splits, args.output_dir, temp=args.temperature, seed=args.seed)
    elif t == "lzw":
        fit_and_apply_lzw(args.vocab_size, train_fit_lines, splits, args.output_dir)
    elif t == "morfessor":
        fit_and_apply_morfessor(args.vocab_size, train_fit_lines, splits, args.output_dir)
    else:
        raise ValueError(f"Unsupported tokenizer: {t}")

    print(f"Pipeline complete in {time.time() - start_time:.1f}s. Files written to {args.output_dir}")


if __name__ == "__main__":
    main()

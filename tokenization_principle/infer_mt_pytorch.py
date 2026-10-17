#!/usr/bin/env python3
"""
Fast PyTorch NMT Inference script.
Generates translations from a trained checkpoint and outputs plain translated text.
"""

import argparse
import os

import torch

from tokenization_principle.train_mt_pytorch import (
    Seq2SeqTransformer,
    Vocabulary,
    greedy_translate,
)


def parse_args():
    parser = argparse.ArgumentParser(description="PyTorch NMT Inference")
    parser.add_argument("--model-path", required=True, help="Path to checkpoint_best.pt")
    parser.add_argument("--vocab-path", default=None, help="Path to vocab.txt (defaults to same dir as checkpoint)")
    parser.add_argument("--input", required=True, help="Path to input source text file")
    parser.add_argument("--output", default=None, help="Path to output file (if None, prints to stdout)")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-len", type=int, default=120)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Locate vocab
    vocab_path = args.vocab_path
    if vocab_path is None or not os.path.exists(vocab_path):
        ckpt_dir = os.path.dirname(os.path.abspath(args.model_path))
        candidate = os.path.join(ckpt_dir, "vocab.txt")
        if os.path.exists(candidate):
            vocab_path = candidate
        else:
            candidate_parent = os.path.join(os.path.dirname(ckpt_dir), "vocab.txt")
            if os.path.exists(candidate_parent):
                vocab_path = candidate_parent
            else:
                raise FileNotFoundError(f"Vocabulary file not found. Checked {vocab_path} and {candidate}")

    vocab = Vocabulary.load(vocab_path)

    # Load model checkpoint
    ckpt = torch.load(args.model_path, map_location=device)
    train_args = ckpt.get("args", {})

    model = Seq2SeqTransformer(
        vocab_size=len(vocab),
        d_model=train_args.get("embed_dim", 512),
        nhead=train_args.get("num_heads", 4),
        num_encoder_layers=train_args.get("encoder_layers", 6),
        num_decoder_layers=train_args.get("decoder_layers", 6),
        dim_feedforward=train_args.get("ffn_dim", 1024),
        dropout=train_args.get("dropout", 0.3),
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Read input lines
    with open(args.input, "r", encoding="utf-8") as f_in:
        src_lines = [l.strip() for l in f_in]

    amp_dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16

    with torch.inference_mode(), torch.amp.autocast("cuda", dtype=amp_dtype, enabled=device.type == "cuda"):
        hypotheses = greedy_translate(
            model,
            src_lines,
            vocab,
            device,
            max_len=args.max_len,
            batch_size=args.batch_size,
        )

    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f_out:
            for hyp in hypotheses:
                f_out.write(f"{hyp}\n")
    else:
        for hyp in hypotheses:
            print(hyp)


if __name__ == "__main__":
    main()

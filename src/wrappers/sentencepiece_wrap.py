#!/usr/bin/env python3

import argparse
import json
import sentencepiece as spm

args = argparse.ArgumentParser()
args.add_argument("-i", "--input", nargs="+", default=["data/CCrawl.de-en/train.tok.en", "data/CCrawl.de-en/train.tok.de"])
args.add_argument("-pi", "--process-input", nargs="+",
    default=[
        "data/CCrawl.de-en/dev.tok.en",
        "data/CCrawl.de-en/dev.tok.de",
        "data/CCrawl.de-en/test.tok.en",
        "data/CCrawl.de-en/test.tok.de",
        "data/CCrawl.de-en/train.tok.en",
        "data/CCrawl.de-en/train.tok.de",
    ])
args.add_argument("-po", "--process-output", nargs="+",
    default=[
        "data/model_spm/dev.tok.en",
        "data/model_spm/dev.tok.de",
        "data/model_spm/test.tok.en",
        "data/model_spm/test.tok.de",
        "data/model_spm/train.en",
        "data/model_spm/train.de",
    ])
args.add_argument("-vo", "--vocab-output", default="data/model_spm/")
args.add_argument("-vs", "--vocab-size", type=int, default=8192)
args.add_argument("-n", "--number-of-lines", type=int, default=100000)
args.add_argument(
    "-pn", "--process-number-of-lines", type=int, nargs="+",
    default=[50000, 50000, 50000, 50000, 1000000, 1000000]
)
# unigram, bpe
args.add_argument("-m", "--model", default="unigram")
args.add_argument("--logfile", default=None)

args = args.parse_args()

print("Fitting model")
spm.SentencePieceTrainer.train(
    input=args.input,
    input_sentence_size=len(args.input)*args.number_of_lines,
    model_prefix=args.vocab_output + args.model,
    vocab_size=args.vocab_size,
    shuffle_input_sentence=True,
    model_type=args.model,
)

sp = spm.SentencePieceProcessor(model_file=args.vocab_output+args.model + ".model")

total_subwords = 0
total_unks = 0

for fname_out, fname_in in zip(args.process_output, args.process_input):
    with open(fname_in, "r") as f:
        data = [x.rstrip("\n") for x in f.readlines()[:args.process_number_of_lines]]
    
    total_words = sum(line.count(" ") + 1 for line in data)
    data = [" ".join(line) for line in sp.encode(data, out_type="str")]
    with open(fname_out, "w") as f:
        for line in data:
            f.write(line + "\n")
        total_subwords = sum(line.count(" ") + 1 for line in data)
        total_unks = sum((" " + line).count("<unk>") for line in data)

        print(fname_in)
        print("Outputting", total_subwords, "total subwords")
        print(
            f"Local otal of {total_unks} UNKs outputted",
            f"({total_unks/total_subwords:.4%} of all subwords)"
        )

    logline = {
        "model": "sentencepiece",
        "method": args.model,
        "vocab_size": args.vocab_size,
        "total_subwords": total_subwords,
        "total_unks": total_unks,
        "total_words": total_words,
        "number_of_lines": len(data),
        "output": fname_out,
        "input": fname_in,
    }
    print(logline)
    if args.logfile is not None:
        with open(args.logfile, "a") as f:
            f.write(json.dumps(logline)+"\n")
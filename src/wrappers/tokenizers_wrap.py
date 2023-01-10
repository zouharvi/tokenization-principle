#!/usr/bin/env python3

import argparse
import json
import tempfile
import tokenizers

args = argparse.ArgumentParser()
args.add_argument(
    "-i", "--input", nargs="+",
    default=[
        "data/CCrawl.de-en/train.tok.en",
        "data/CCrawl.de-en/train.tok.de"
    ]
)
args.add_argument(
    "-vo", "--vocab-output",
    default="data/model_tokenizer/model.json"
)
args.add_argument(
    "-pi", "--process-input", nargs="+",
    default=[
        "data/CCrawl.de-en/dev.tok.en",
        "data/CCrawl.de-en/dev.tok.de",
        "data/CCrawl.de-en/test.tok.en",
        "data/CCrawl.de-en/test.tok.de",
        "data/CCrawl.de-en/train.tok.en",
        "data/CCrawl.de-en/train.tok.de",
    ])
args.add_argument(
    "-po", "--process-output", nargs="+",
    default=[
        "data/model_tokenizer_MODEL/dev.tok.en",
        "data/model_tokenizer_MODEL/dev.tok.de",
        "data/model_tokenizer_MODEL/test.tok.en",
        "data/model_tokenizer_MODEL/test.tok.de",
        "data/model_tokenizer_MODEL/train.en",
        "data/model_tokenizer_MODEL/train.de",
    ])
args.add_argument(
    "-pn", "--process-number-of-lines", type=int, nargs="+",
    default=[50000, 50000, 50000, 50000, 1000000, 1000000]
)
args.add_argument("-vs", "--vocab-size", type=int, default=8000)
args.add_argument("-n", "--number-of-lines", type=int, default=100000)
# unigram, wordlevel, wordpiece
args.add_argument("--model", default="unigram")
args.add_argument("--logfile", default=None)
args = args.parse_args()


def get_trainer(name):
    if name == "unigram":
        return tokenizers.trainers.UnigramTrainer
    elif name == "wordlevel":
        return tokenizers.trainers.WordLevelTrainer
    elif name == "wordpiece":
        return tokenizers.trainers.WordPieceTrainer
    elif name == "bpe":
        return tokenizers.trainers.BpeTrainer


def get_tokenizer(name):
    if name == "unigram":
        return tokenizers.models.Unigram
    elif name == "wordlevel":
        return tokenizers.models.WordLevel
    elif name == "wordpiece":
        return tokenizers.models.WordPiece
    elif name == "bpe":
        return tokenizers.models.BPE


with tempfile.NamedTemporaryFile() as fname:
    fname = fname.name
    with open(fname, "w") as f:
        for input_fname in args.input:
            data = open(input_fname, "r").readlines()[:args.number_of_lines]
            f.writelines(data)

    # weird inconsistency
    # https://github.com/huggingface/tokenizers/issues/586
    tokenizer_model = get_tokenizer(args.model)()
    if args.model in {"unigram"}:
        trainer = get_trainer(args.model)(
            unk_token="[UNK]",
            vocab_size=args.vocab_size, special_tokens=["[UNK]"]
        )
    else:
        trainer = get_trainer(args.model)(
            vocab_size=args.vocab_size, special_tokens=["[UNK]"]
        )
        tokenizer_model.unk_token = "[UNK]"
    tokenizer = tokenizers.Tokenizer(tokenizer_model)
    tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.WhitespaceSplit()

    tokenizer.train([fname], trainer)
    tokenizer.save(args.vocab_output)

for fname_out, fname_in, process_number_of_lines in zip(
    args.process_output,
    args.process_input,
    args.process_number_of_lines
):
    total_subwords = 0
    total_unks = 0

    with open(fname_in, "r") as f:
        data_in = [
            x.rstrip("\n")
            for x in f.readlines()[:process_number_of_lines]
        ]

    total_words = sum(line.count(" ") + 1 for line in data_in)
    data = tokenizer.encode_batch(data_in)
    print(data[0].tokens)
    with open(fname_out, "w") as f:
        total_subwords += sum(len(line.tokens) for line in data)
        for line_in, line in zip(data_in, data):
            last_right = None
            tokens = []
            for token, offset in zip(line.tokens, line.offsets):
                token = token.removeprefix("##")
                # starts matching
                if last_right != offset[0]:
                    tokens.append(token)
                else:
                    tokens.append("@@"+token)
                last_right = offset[1]
            # replace direction of unks
            line = " ".join(tokens).replace(" @@", "@@ ")
            total_unks += line.count("[UNK]")
            f.write(line + "\n")
        print(data[-1].tokens)
        print(line)

        print(fname_in)
        print("Outputting", total_subwords, "total subwords")
        print(
            f"Local total of {total_unks} UNKs outputted",
            f"({total_unks/total_subwords:.4%} of all subwords)"
        )

    logline = {
        "model": "tokenizers",
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
            f.write(json.dumps(logline) + "\n")

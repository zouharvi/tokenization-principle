#!/usr/bin/env python3

import glob
import collections

data = {
    "cs-en": collections.defaultdict([]),
    "de-en": collections.defaultdict([]),
}

first = False
for f in glob.glob("data_bin_ckpt_repl/*/model_bpe_random/*/dev_out.en"):
    pass

for f in glob.glob("data_bin_ckpt_repl/*/model_bpe_random/*/dev_out.en"):
    fs = f.split("/")
    lang = fs[1]
    signature = fs[-2]
    print(lang, signature)
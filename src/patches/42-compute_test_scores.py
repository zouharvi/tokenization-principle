#!/usr/bin/env python3

import glob
import evalu

data = {
    "cs-en": {},
    "de-en": {},
}

for f in glob.glob("data_bin_ckpt_repl/*/model_bpe_random/*/dev_out.en"):
    print(f)
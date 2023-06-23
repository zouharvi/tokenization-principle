#!/usr/bin/env python3

import pickle
import glob
import pandas as pd

data_all = []
for fname in glob.glob("computed/bleu_corr_cache_multi_processed_*.pkl"):
    power = float(fname.split("/")[-1].removeprefix("bleu_corr_cache_multi_processed_").removesuffix(".pkl"))
    data_local = pickle.load(open(fname, "rb"))
    
    for tokenizer_name, tokenizer_data in data_local:
        for tokenizer_config, (val_pred, val_bleu) in tokenizer_data.items():
            data_all.append((
                power, tokenizer_name, val_pred, val_bleu
            ))
            print(power, tokenizer_name, val_pred, val_bleu)

data_all = pd.DataFrame(data_all, columns=["alpha", "tokenizer", "pred", "bleu"])
data_all.to_csv("computed/experiment2.csv", index=False)

data_all = pd.read_csv("computed/experiment2.csv")
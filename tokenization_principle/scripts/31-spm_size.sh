#!/usr/bin/bash

python3 -m tokenization_principle.wrappers.tokenizers_wrap \
    --vocab-output /dev/null \
    --model bpe \
    --vocab-size 4000 \
    --number-of-lines 10000 \
    --process-number-of-lines 10000 \
    --logfile /dev/null \
    --input data/flattened/n3_e10_p01_k3/dev.en \
    --process-input data/flattened/n3_e10_p01_k3/dev.en \
    --process-output /dev/null \
;


python3 -m tokenization_principle.wrappers.tokenizers_wrap \
    --vocab-output /dev/null \
    --model bpe \
    --vocab-size 4000 \
    --number-of-lines 10000 \
    --process-number-of-lines 10000 \
    --logfile /dev/null \
    --input data/CCrawl.de-en/dev.tok.en \
    --process-input data/CCrawl.de-en/dev.tok.en \
    --process-output /dev/null \
;

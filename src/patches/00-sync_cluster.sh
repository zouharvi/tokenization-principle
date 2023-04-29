#!/usr/bin/bash

rsync -azP --filter=":- .gitignore" --exclude .git/ . euler:/cluster/work/sachan/vilem/tokenization-principle/

# scp euler:/cluster/work/sachan/vilem/random-bpe/computed/* computed/
# scp euler:/cluster/work/sachan/vilem/random-bpe/computed/apply_bpe_all.jsonl computed/
# scp euler:/cluster/work/sachan/vilem/random-bpe/logs/train_mt_*.log logs
# scp euler:/cluster/work/sachan/vilem/random-bpe/computed/glabrus.jsonl computed/
# rsync -azP euler:/cluster/work/sachan/vilem/random-bpe/data_tmp/* data/
# rsync -azP euler:/cluster/work/sachan/vilem/random-bpe/logs/train_mt_s*.log logs/
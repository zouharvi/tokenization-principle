#!/usr/bin/bash

rsync -azP --filter=":- .gitignore" --exclude .git/ . euler:/cluster/work/sachan/vilem/predicting-performance/

# rsync -azP euler:/cluster/work/sachan/vilem/predicting-performance/logs/train_mt_s*.log logs/
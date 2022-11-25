#!/usr/bin/bash

rsync -azP --filter=":- .gitignore" --exclude .git/ . euler:/cluster/work/sachan/vilem/random-bpe/

# scp euler:/cluster/work/sachan/vilem/random-bpe/computed/* computed/
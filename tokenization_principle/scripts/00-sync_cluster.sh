#!/usr/bin/bash

rsync -azP --filter=":- .gitignore" --exclude .git/ . euler:/cluster/work/sachan/vilem/tokenization-principle/

# scp euler:/cluster/work/sachan/vilem/tokenization-principle/results.toml ~/Downloads/

# scp euler:/cluster/work/sachan/vilem/random-bpe/computed/* computed/
# scp euler:/cluster/work/sachan/vilem/random-bpe/computed/apply_bpe_all.jsonl computed/
# scp euler:/cluster/work/sachan/vilem/random-bpe/logs/train_mt_*.log logs
# scp euler:/cluster/work/sachan/vilem/random-bpe/computed/glabrus.jsonl computed/
# rsync -azP euler:/cluster/work/sachan/vilem/random-bpe/data_tmp/* data/
# rsync -azP euler:/cluster/work/sachan/vilem/random-bpe/logs/train_mt_s*.log logs/

# scp euler:/cluster/work/sachan/vilem/tokenization-principle/data/model_bpe_random/cs-en/005_16k/dev.en data/dev.bpe.en
# scp euler:/cluster/work/sachan/vilem/tokenization-principle/computed/metric_scores_repl* computed/


# sbatch --time=1-0 --mem-per-cpu=20G --gpus=1 \
#     -J "toml_de-en_lzw_v16000" \
#     --output="logs/toml_de-en_lzw_v16000.log" \
#     --wrap="python3 -u tokenization_principle/scripts/06-compute_toml.py \
#         --dir data/prepared/de-en/lzw_v16000 \
#         --overwrite"

# sbatch --time=1-0 --mem-per-cpu=20G --gpus=1 \
#     -J "toml_de-en_lzw_v8000" \
#     --output="logs/toml_de-en_lzw_v8000.log" \
#     --wrap="python3 -u tokenization_principle/scripts/06-compute_toml.py \
#         --dir data/prepared/de-en/lzw_v8000 \
#         --overwrite"

# sbatch --time=1-0 --ntasks=10 --mem-per-cpu=16G \
#     -J "prep_zh-en_unigram_v4000" \
#     --output="logs/prep_zh-en_unigram_v4000.log" \
#     --wrap="python3 -m tokenization_principle.prepare_data --tokenizer unigram --vocab-size 6000 --lang-pair zh-en --train-lines 10000000 --input-dir data/raw/zh-en --output-dir data/prepared/zh-en/unigram_v4000"   
  
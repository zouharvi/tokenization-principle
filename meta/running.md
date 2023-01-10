|start date|status|nickname|comment|command|
|-|-|-|-|-|
|10-01-2023|running|paten|train MT|`./src/patches/24-train_mt.sh`|
|10-01-2023|ok|paten|train & apply tokenizer 16k, 8k, 4k|`./src/patches/20-submit_train_all.sh`|
|01-01-2023|running|kinetics (glabrus)|train MT, seeds 1, 2, 3, 4, 5, amp|`./src/patches/28-train_mt_random_only.sh`|
|31-12-2022|ok|glabrus|train MT, seed 2, amp|`./src/patches/28-train_mt_random_only.sh`|
|29-12-2022|ok|glabrus|train MT|`./src/patches/28-train_mt_random_only.sh`|
|29-12-2022|ok|glabrus|apply tokenizers|`./src/patches/26-submit_apply_random_only.sh`|
|29-12-2022|ok|glabrus|train tokenizers|`./src/patches/25-submit_train_random_only.sh`|
|24-12-2022|ok|bouree|train tokenizers|`./src/patches/20-submit_train_all.sh`|
|24-12-2022|ok|azaroth|train tokenizers|`./src/patches/21-submit_apply_all.sh`|
|24-12-2022|ok|obturate|train tokenizers|`./src/patches/20-submit_train_all.sh`|
|29-11-2022|ok||beam training|`./src/patches/13-submit_train_bpe_beam.sh`|
|29-11-2022|ok||beam training|`./src/patches/12-submit_bpe_beam.sh`|
|29-11-2022|ok||BPE train & apply small|`./src/patches/09-train_small_bpe.sh`|
|27-11-2022|ok||apply BPE large (faster)|`./src/patches/05-submit_apply_bpe_large.sh`|
|26-11-2022|ok||MT training|`./src/patches/07-train_mt.sh`|
|26-11-2022|ok||preprocess data|`./src/patches/06-preprocess_data.sh`|
|26-11-2022|ok||apply BPE large|`./src/patches/05-submit_apply_bpe_large.sh`|
|25-11-2022|ok||apply BPE|`./src/patches/03-submit_apply_bpe.sh`|
|25-11-2022|ok||fit BPE on 100k|`./src/patches/02-submit_fit_bpe.sh`|
|25-11-2022|ok||tokenize|`./src/patches/04-tokenize_data.sh`|
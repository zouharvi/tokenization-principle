|start date|status|nickname|comment|command|
|-|-|-|-|-|
|29-11-2022|running||beam training|`./src/patches/12-submit_bpe_beam.sh`|
|29-11-2022|ok||BPE train & apply small|`./src/patches/09-train_small_bpe.sh`|
|27-11-2022|ok||apply BPE large (faster)|`./src/patches/05-submit_apply_bpe_large.sh`|
|26-11-2022|running||MT training|`./src/patches/07-train_mt.sh`|
|26-11-2022|ok||preprocess data|`./src/patches/06-preprocess_data.sh`|
|26-11-2022|ok||apply BPE large|`./src/patches/05-submit_apply_bpe_large.sh`|
|25-11-2022|ok||apply BPE|`./src/patches/03-submit_apply_bpe.sh`|
|25-11-2022|ok||fit BPE on 100k|`./src/patches/02-submit_fit_bpe.sh`|
|25-11-2022|ok||tokenize|`./src/patches/04-tokenize_data.sh`|
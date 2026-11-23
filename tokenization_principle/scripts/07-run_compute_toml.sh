for MODEL_DIR in data/prepared/*/*; do
    [ -d "${MODEL_DIR}" ] || continue
    # Skip if output.toml already exists (06-compute_toml.py also skips without --overwrite)
    [ -f "${MODEL_DIR}/output.toml" ] && continue
    # Skip if MT test hypotheses don't exist yet
    compgen -G "${MODEL_DIR}/checkpoints/test_s*.txt" > /dev/null || continue

    LANGS=$(basename "$(dirname "${MODEL_DIR}")")
    SIG=$(basename "${MODEL_DIR}")

    echo "Submitting ${LANGS} ${SIG}"
    sbatch --time=1-0 --mem-per-cpu=40G --gpus=1 \
        -J "toml_${LANGS}_${SIG}" \
        --output="logs/toml_${LANGS}_${SIG}.log" \
        --wrap="python3 -u tokenization_principle/scripts/06-compute_toml.py --dir \"${MODEL_DIR}\""
done
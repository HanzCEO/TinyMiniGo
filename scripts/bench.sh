#!/usr/bin/env bash
# Benchmark TinyMiniGo on the reference MiniCPM5-1B command.
# Usage: scripts/bench.sh [MAX_TOKENS] [PROMPT]
set -euo pipefail
MAX_TOKENS="${1:-64}"
PROMPT="${2:-Who are you?}"
MODEL="${TINYMINIGO_MODEL:-$HOME/Documents/Models/openbmb/MiniCPM5-1B/model-00000-of-00001.safetensors}"
TEMPLATE="${TINYMINIGO_TEMPLATE:-$HOME/Documents/Models/openbmb/MiniCPM5-1B/chat_template.jinja}"

# Force rebuild (mtime skew on this system can leave stale binaries)
touch src/main.rs src/model.rs src/tensor.rs
sleep 1.1
cargo build -r --quiet
echo "=== bench: max_tokens=$MAX_TOKENS prompt=\"$PROMPT\" ==="
START=$(date +%s.%N)
./target/release/tinyminigo -m "$MODEL" --template "$TEMPLATE" \
    --max-tokens "$MAX_TOKENS" "$PROMPT" > /tmp/bench_stdout.txt
END=$(date +%s.%N)
echo "wall=$(echo "$END $START" | awk '{printf "%.2f", $1-$2}')s"
echo "--- output ---"
cat /tmp/bench_stdout.txt
echo "--------------"
md5sum /tmp/bench_stdout.txt

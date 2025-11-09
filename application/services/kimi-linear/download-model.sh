#!/bin/bash
set -e

MODEL_NAME="${MODEL_NAME:-moonshotai/Kimi-Linear-48B-A3B-Instruct}"
MODEL_PATH="${MODEL_PATH:-/models/kimi-linear-48b}"

echo "🚀 Lade Kimi Linear 48B von ${MODEL_NAME}..."

# Installiere Git LFS
if ! command -v git-lfs &>/dev/null; then
    apt-get update && apt-get install -y git-lfs
fi
git lfs install

# Erstelle Verzeichnis
mkdir -p "$MODEL_PATH"
cd "$MODEL_PATH"

# Lade mit Git LFS (nur wichtige Dateien)
echo "📥 Klone Repository (dies kann 10-20 Minuten dauern)..."

if [[ -n "${HF_TOKEN:-}" ]]; then
    git clone --depth 1 --filter=blob:none --sparse \
        "https://user:${HF_TOKEN}@huggingface.co/${MODEL_NAME}" .
else
    git clone --depth 1 --filter=blob:none --sparse \
        "https://huggingface.co/${MODEL_NAME}" .
fi

# Sparse checkout für wichtige Dateien
git sparse-checkout init --cone
git sparse-checkout set \
    "*.json" "*.py" "*.md" "*.txt" "tokenizer.*" \
    "*.safetensors" "model-*.safetensors" "*.bin" "config.json"

echo "✅ Modell-Download abgeschlossen"
echo "📊 Speicherplatz:"
du -sh .
echo "📁 Dateien:"
ls -lh
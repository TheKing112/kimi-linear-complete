#!/bin/bash
set -euo pipefail

MODEL_NAME="${MODEL_NAME:-moonshotai/Kimi-Linear-48B-A3B-Instruct}"
MODEL_PATH="${MODEL_PATH:-/models/kimi-linear-48b}"
REQUIRED_SPACE_GB=100

# Speicherplatz-Check
echo "🔍 Prüfe Speicherplatz..."
if [ ! -d "$MODEL_PATH" ]; then
    mkdir -p "$MODEL_PATH"
fi

AVAILABLE_SPACE_GB=$(df -BG "$MODEL_PATH" | tail -1 | awk '{print $4}' | sed 's/G//')
if [ "${AVAILABLE_SPACE_GB:-0}" -lt "$REQUIRED_SPACE_GB" ]; then
    echo "❌ Nicht genug Speicherplatz!"
    echo "   Benötigt: ${REQUIRED_SPACE_GB}GB"
    echo "   Verfügbar: ${AVAILABLE_SPACE_GB}GB"
    exit 1
fi

echo "🚀 Lade Kimi Linear 48B von ${MODEL_NAME}..."

# Git LFS Installation (systemunabhängiger)
if ! command -v git-lfs &>/dev/null; then
    if command -v apt-get &>/dev/null; then
        apt-get update && apt-get install -y git-lfs
    elif command -v yum &>/dev/null; then
        yum install -y git-lfs
    elif command -v brew &>/dev/null; then
        brew install git-lfs
    else
        echo "❌ Git LFS konnte nicht installiert werden. Bitte manuell installieren."
        exit 1
    fi
fi
git lfs install

# Bereinige falls vorhanden
if [ -d "${MODEL_PATH}/.git" ]; then
    echo "⚠️  Bereinige bestehendes Repository..."
    rm -rf "${MODEL_PATH:?}"/*
fi

cd "$MODEL_PATH"

# Repository klonen
echo "📥 Klone Repository (dies kann 10-20 Minuten dauern)..."
if ! timeout 1800 git clone --depth 1 --filter=blob:none --sparse \
    "https://huggingface.co/${MODEL_NAME}" . ; then
    echo "⚠️  Clone fehlgeschlagen, versuche Fallback..."
    
    if command -v huggingface-cli &>/dev/null; then
        huggingface-cli download "${MODEL_NAME}" --local-dir .
    else
        echo "❌ Keine alternative Download-Methode verfügbar"
        echo "Installieren Sie: pip install huggingface_hub"
        exit 1
    fi
fi

# Sparse checkout für wichtige Dateien (bereinigt)
echo "📦 Lade Modell-Dateien..."
git sparse-checkout init --cone
git sparse-checkout set \
    "*.json" "*.py" "tokenizer.*" "*.safetensors" "model-*.safetensors" "*.bin"

echo "✅ Download abgeschlossen"
echo "📊 Speicherplatz: $(du -sh . 2>/dev/null || echo 'N/A')"
echo "📁 Dateien: $(ls -lh | wc -l) Dateien gefunden"

### ✅ KORRIGIERTER VERIFIZIERUNGSBLOCK
echo "✅ Verifiziere Download..."

# Wichtige Konfigurationsdateien (erweiterte Liste)
REQUIRED_FILES=(
    "config.json"
    "tokenizer_config.json"
    "tokenizer.json"
    "preprocessor_config.json"
    "model_index.json"
)

# Prüfe Existenz und minimale Größe (nicht leer)
for file in "${REQUIRED_FILES[@]}"; do
    if [ -f "$file" ]; then
        if [ ! -s "$file" ]; then
            echo "❌ Datei ist leer: $file"
            exit 1
        fi
    fi
done

# Prüfe Model-Weights (mindestens eine nicht-leere .safetensors Datei)
if ! find . -maxdepth 1 -name "*.safetensors" -type f -size +0 2>/dev/null | grep -q .; then
    echo "❌ Keine validen Model-Weights gefunden!"
    exit 1
fi

# Optionale aber empfohlene Prüfung: Modell-Konsistenz
if [ -f "config.json" ] && command -v python3 &>/dev/null; then
    python3 -c "import json; json.load(open('config.json'))" 2>/dev/null || \
        echo "⚠️ Warnung: config.json ist keine gültige JSON-Datei"
fi

echo "✅ Download und Verifikation erfolgreich"
echo "🎯 Modell bereit unter: ${MODEL_PATH}"
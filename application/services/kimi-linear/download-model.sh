#!/bin/bash
set -e

MODEL_NAME="${MODEL_NAME:-moonshotai/Kimi-Linear-48B-A3B-Instruct}"
MODEL_PATH="${MODEL_PATH:-/models/kimi-linear-48b}"

# ✅ NEU: Speicherplatz-Check am Anfang
echo "🔍 Prüfe Festplattenplatz..."
REQUIRED_SPACE_GB=100
AVAILABLE_SPACE_GB=$(df -BG "$MODEL_PATH" | tail -1 | awk '{print $4}' | sed 's/G//')

if [ "$AVAILABLE_SPACE_GB" -lt "$REQUIRED_SPACE_GB" ]; then
    echo "❌ Nicht genug Speicherplatz!"
    echo "   Benötigt: ${REQUIRED_SPACE_GB}GB"
    echo "   Verfügbar: ${AVAILABLE_SPACE_GB}GB"
    exit 1
fi

echo "🚀 Lade Kimi Linear 48B von ${MODEL_NAME}..."

# Installiere Git LFS
if ! command -v git-lfs &>/dev/null; then
    apt-get update && apt-get install -y git-lfs
fi
git lfs install

# Erstelle Verzeichnis
mkdir -p "$MODEL_PATH"
cd "$MODEL_PATH"

# ✅ KORRIGIERT: Fehlerbehandlung und Fallback hinzugefügt
echo "📥 Klone Repository (dies kann 10-20 Minuten dauern)..."
if ! git clone --depth 1 --filter=blob:none --sparse \
    "https://huggingface.co/${MODEL_NAME}" . ; then
    echo "❌ Clone fehlgeschlagen!"
    echo "Versuche alternative Methode..."
    
    # Fallback: Download via huggingface-cli
    if command -v huggingface-cli &>/dev/null; then
        huggingface-cli download "${MODEL_NAME}" --local-dir .
    else
        echo "❌ Keine alternative Download-Methode verfügbar"
        exit 1
    fi
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

# ✅ NEU: Verifizierung am Ende
echo "✅ Verifiziere Download..."

# Check for essential files
REQUIRED_FILES=("config.json" "tokenizer_config.json")
for file in "${REQUIRED_FILES[@]}"; do
    if [ ! -f "$file" ]; then
        echo "❌ Fehlende Datei: $file"
        exit 1
    fi
done

# Check for model weights (mindestens eine .safetensors Datei)
if ! ls *.safetensors 1> /dev/null 2>&1; then
    echo "❌ Keine Model-Weights gefunden!"
    exit 1
fi

echo "✅ Download und Verifikation erfolgreich"
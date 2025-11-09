FROM nvcr.io/nvidia/pytorch:24.02-py3  

# Build-Argumente
ARG MODEL_NAME="moonshotai/Kimi-Linear-48B-A3B-Instruct"
ARG MODEL_PATH="/models/kimi-linear-48b"

# Umgebungsvariablen
ENV TZ=Europe/Berlin \
    MODEL_NAME=${MODEL_NAME} \
    MODEL_PATH=${MODEL_PATH} \
    DEVICE=cuda \
    LOAD_IN_4BIT=true \
    TORCH_COMPILE=true \
    TRUST_REMOTE_CODE=true \
    MAX_INPUT_LENGTH=1048576 \
    PORT=8003 \
    CUDA_VISIBLE_DEVICES=0 \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    PYTHONUNBUFFERED=1

# System-Pakete
RUN apt-get update && apt-get install -y \
    git git-lfs curl wget build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ✅ Requirements ZUERST kopieren (besseres Caching)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# ✅ DANN Code kopieren
COPY src/ ./src/
COPY download-model.sh ./
RUN chmod +x download-model.sh

# Modell-Verzeichnis
RUN mkdir -p ${MODEL_PATH}

# Modell herunterladen
RUN bash download-model.sh

# ✅ Healthcheck verbessert
HEALTHCHECK --interval=60s --timeout=30s --start-period=1200s --retries=5 \
    CMD curl -f http://localhost:8003/health || exit 1

ENTRYPOINT ["python", "-m", "uvicorn", "src.server:app", "--host", "0.0.0.0", "--port", "8003", "--workers", "1"]
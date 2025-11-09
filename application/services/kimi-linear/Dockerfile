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
    HF_HUB_ENABLE_HF_TRANSFER=1  
# System-Pakete
RUN apt-get update && apt-get install -y \
    git git-lfs curl wget build-essential \
    && rm -rf /var/lib/apt/lists/*

# Installiere PyTorch und FLA-Core (Sparse Activation)
RUN pip install --no-cache-dir --upgrade pip setuptools wheel
RUN pip install --no-cache-dir torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
RUN pip install --no-cache-dir fla-core==0.4.0  # Wichtig für A3B!

# Haupt-Dependencies
RUN pip install --no-cache-dir \
    transformers==4.49.0 \
    accelerate==1.2.1 \
    bitsandbytes==0.45.0 \
    scipy==1.13.1 \
    fastapi==0.115.0 \
    uvicorn[standard]==0.34.0 \
    pydantic==2.10.0 \
    psutil==5.9.8

# Modell-Verzeichnis
RUN mkdir -p ${MODEL_PATH}

# Service-Code kopieren
COPY src/ ./src/
COPY download-model.sh ./

# Modell herunterladen (mit Git LFS)
RUN bash download-model.sh

# Healthcheck
HEALTHCHECK --interval=60s --timeout=30s --start-period=300s --retries=5 \
    CMD curl -f http://localhost:8003/health || exit 1

# Start
ENTRYPOINT ["python", "-m", "uvicorn", "src.server:app", "--host", "0.0.0.0", "--port", "8003", "--workers", "1"]
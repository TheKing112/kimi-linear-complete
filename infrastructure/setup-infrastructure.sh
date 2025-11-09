#!/bin/bash
# =============================================================================
# KIMI LINEAR 48B - INFRASTRUCTURE SETUP (SPARSE ACTIVATION)
# 48B Parameter, aber nur 3B aktiv pro Forward-Pass!
# =============================================================================

set -euo pipefail

# ==================================== KONFIGURATION ====================================

readonly NVIDIA_VERSION="535"
readonly CUDA_VERSION="12-1"
readonly PYTHON_VERSION="3.11"
readonly MIN_VRAM_GB="8"
readonly LOG_FILE="/tmp/infrastructure-setup.log"

# ==================================== HELPER FUNCTIONS ====================================

update_system() {
    log "Updating system packages..."
    sudo apt update && sudo apt upgrade -y
}

install_essentials() {
    log "Installing essential packages..."
    sudo apt install -y curl git build-essential software-properties-common wget apt-transport-https ca-certificates
}

install_docker() {
    log "Installing Docker..."
    curl -fsSL https://get.docker.com | sh
    sudo usermod -aG docker $USER
}

install_nvidia() {
    log "Installing NVIDIA drivers..."
    sudo apt install -y nvidia-driver-$NVIDIA_VERSION nvidia-dkms-$NVIDIA_VERSION
}

install_nvidia_docker() {
    log "Installing NVIDIA Docker runtime..."
    distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
    curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
    curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
    sudo apt update && sudo apt install -y nvidia-docker2
    sudo systemctl restart docker
}

create_directories() {
    log "Creating application directories..."
    mkdir -p /home/ubuntu/kimi-linear-complete/{application/services,models,logs}
    mkdir -p ~/.cache/huggingface
}

test_environment() {
    log "Testing environment..."
    nvidia-smi || warn "GPU not detected or nvidia-smi not available"
    docker --version || error "Docker not installed correctly"
    python3 --version || error "Python3 not available"
}

# ==================================== LOGGING ====================================

log() { echo "[$(date +%T)] $1" | tee -a "$LOG_FILE"; }
ok()  { echo "✓ $1" | tee -a "$LOG_FILE"; }
error() { echo "✗ $1" | tee -a "$LOG_FILE"; exit 1; }
warn() { echo "⚠ $1" | tee -a "$LOG_FILE"; }

# ==================================== SYSTEM-SETUP ====================================

check_gpu() {
    if lspci | grep -i nvidia &>/dev/null; then
        echo "true"
    else
        echo "false"
    fi
}

check_vram() {
    if command -v nvidia-smi &>/dev/null; then
        local total_vram=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
        total_vram_gb=$((total_vram / 1024))
        if [ "$total_vram_gb" -lt "$MIN_VRAM_GB" ]; then
            echo "insufficient"
        else
            echo "sufficient"
        fi
    else
        echo "unknown"
    fi
}

install_python() {
    log "Installiere Python $PYTHON_VERSION mit Entwicklungs-Tools..."
    
    sudo apt install -y software-properties-common
    sudo add-apt-repository -y ppa:deadsnakes/ppa
    sudo apt update
    
    sudo apt install -y \
        python${PYTHON_VERSION} \
        python${PYTHON_VERSION}-dev \
        python${PYTHON_VERSION}-venv \
        python3-pip
    
    sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python${PYTHON_VERSION} 1
    
    pip3 install --upgrade pip setuptools wheel
    ok "Python $PYTHON_VERSION bereit"
}

# ==================================== NVIDIA & CUDA ====================================

install_cuda() {
    local has_gpu=$(check_gpu)
    if [[ "$has_gpu" == "false" ]]; then
        log "Keine GPU erkannt, überspringe CUDA"
        return 0
    fi

    local vram_status=$(check_vram)
    if [[ "$vram_status" == "insufficient" ]]; then
        warn "WARNUNG: Weniger als ${MIN_VRAM_GB}GB VRAM verfügbar!"
        warn "Modell benötigt ~${MIN_VRAM_GB}GB in 4-bit"
        warn "Fahr mit CPU-Modus fort..."
        return 0
    fi

    log "Installiere CUDA Toolkit $CUDA_VERSION für PyTorch 2.2.x..."
    
    wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
    sudo dpkg -i cuda-keyring_1.1-1_all.deb
    rm cuda-keyring_1.1-1_all.deb
    
    sudo apt update
    sudo apt install -y cuda-toolkit-${CUDA_VERSION//./-}
    
    local cuda_path="/usr/local/cuda-${CUDA_VERSION}"
    if [[ -f "$cuda_path/bin/nvcc" ]]; then
        echo "export PATH=$cuda_path/bin:\$PATH" >> ~/.bashrc
        echo "export LD_LIBRARY_PATH=$cuda_path/lib64:\$LD_LIBRARY_PATH" >> ~/.bashrc
        ok "CUDA $CUDA_VERSION installiert"
    else
        error "CUDA-Installation fehlgeschlagen"
    fi
}

# ==================================== PYTHON ML STACK (KOMPATIBEL) ====================================

install_ml_packages() {
    log "Installiere Python ML Stack mit kompatiblen Versionen..."
    
    python3 -m venv ~/.venv/kimi-linear
    source ~/.venv/kimi-linear/bin/activate
    
    pip install --no-cache-dir torch==2.2.2 --index-url https://download.pytorch.org/whl/cu121
    
    pip install --no-cache-dir fla-core==0.4.0
    
    pip install --no-cache-dir \
        transformers==4.38.2 \
        accelerate==0.27.2 \
        bitsandbytes==0.43.1 \
        scipy==1.13.1 \
        sentence-transformers==3.0.1
    
    pip install --no-cache-dir flash-attn==2.5.6
    
    pip install --no-cache-dir \
        fastapi==0.115.0 \
        uvicorn[standard]==0.34.0 \
        pydantic==2.10.0
    
    pip install --no-cache-dir psutil==5.9.8
    
    ok "ML Stack mit kompatiblen Versionen installiert"
}

# ==================================== HAUPTFUNKTION ====================================

main() {
    log "════════════════════════════════════════════════════════════"
    log "  Kimi Linear 48B - Sparse Activation Setup"
    log "  48B Parameters, 3B Active per Forward Pass"
    log "════════════════════════════════════════════════════════════"
    
    update_system
    install_essentials
    install_python
    install_docker
    
    local has_gpu=$(check_gpu)
    if [[ "$has_gpu" == "true" ]]; then
        install_nvidia
        install_cuda
        install_nvidia_docker
        
        local vram_status=$(check_vram)
        if [[ "$vram_status" == "insufficient" ]]; then
            warn "════════════════════════════════════════════════════════════"
            warn "  ⚠️  VRAM-WARNUNG: Modell benötigt ${MIN_VRAM_GB}GB+"
            warn "  Bitte verwende GPUs mit ausreichend Speicher!"
            warn "════════════════════════════════════════════════════════════"
        fi
    else
        warn "════════════════════════════════════════════════════════════"
        warn "  ⚠️  KEINE GPU ERKANNT - CPU-Modus wird NICHT empfohlen!"
        warn "  Ladezeit: ~30-60 Minuten, sehr langsame Inferenz"
        warn "════════════════════════════════════════════════════════════"
    fi
    
    install_ml_packages
    create_directories
    test_environment
    
    log "════════════════════════════════════════════════════════════"
    log "  INFRASTRUKTUR BEREIT! Nur ${MIN_VRAM_GB}GB VRAM erforderlich!"
    log "  Sparse Activation: 48B total → 3B active"
    log "════════════════════════════════════════════════════════════"
}

main "$@" 2>&1 | tee "$LOG_FILE"
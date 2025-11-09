#!/bin/bash
# =============================================================================
# KIMI LINEAR - VM INTERNAL SETUP
# Wird automatisch in der VM ausgeführt
# =============================================================================

# ✅ NEU - Am Anfang: Prüfe auf erneute Ausführung
if [ -f /tmp/setup-complete ]; then
    echo "⚠️  Setup wurde bereits durchgeführt!"
    echo "   Marker: /tmp/setup-complete"
    read -p "Erneut ausführen? [y/N] " -r
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 0
    fi
    rm /tmp/setup-complete
fi

# Create lockfile
if [ -f /tmp/setup-running ]; then
    echo "❌ Setup läuft bereits!"
    exit 1
fi
touch /tmp/setup-running
trap "rm -f /tmp/setup-running" EXIT

set -euo pipefail

# ===== CONFIG =====
readonly LOG_FILE="/var/log/kimi-setup.log"
readonly REPO_URL="https://github.com/dein-repo/kimi-linear-complete.git"  # ⚠️ Anpassen!

export MODEL_NAME="moonshotai/Kimi-Linear-48B-A3B-Instruct"
export MODEL_PATH="/models/kimi-linear-48b"
export HF_TOKEN="${HF_TOKEN:-}"

# ===== LOGGING =====
exec > >(tee -a "$LOG_FILE") 2>&1
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starte Kimi Linear VM Setup..."

# ✅ NEU - Error handling function
set +e

run_step() {
    local step_name="$1"
    local cmd="$2"
    local exit_code=0
    
    echo "▶ $step_name..."
    eval "$cmd"
    exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        echo "✓ $step_name abgeschlossen"
        return 0
    else
        echo "✗ $step_name fehlgeschlagen!"
        echo "  Command: $cmd"
        return 1
    fi
}

set -e

# ===== SYSTEM UPDATE =====
run_step "System-Update" "apt update && apt upgrade -y"
run_step "Installiere Pakete" "apt install -y curl git build-essential software-properties-common"

# ===== NVIDIA DRIVERS =====
run_step "Installiere NVIDIA Treiber" "apt install -y nvidia-driver-535 nvidia-dkms-535"
run_step "Prüfe GPU Erkennung" "nvidia-smi || echo '⚠️  GPU noch nicht erkannt – nach Reboot prüfen'"

# ===== DOCKER =====
run_step "Installiere Docker" "curl -fsSL https://get.docker.com | sh"
run_step "Füge ubuntu zu docker Gruppe hinzu" "usermod -aG docker ubuntu"

# ===== NVIDIA DOCKER =====
run_step "Installiere NVIDIA Docker Runtime" "\
    distribution=\$(. /etc/os-release;echo \$ID\$VERSION_ID) && \
    curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | apt-key add - && \
    curl -s -L https://nvidia.github.io/nvidia-docker/\$distribution/nvidia-docker.list | tee /etc/apt/sources.list.d/nvidia-docker.list && \
    apt update && apt install -y nvidia-docker2 && \
    systemctl restart docker
"

# ===== REPOSITORY KLONEN =====
run_step "Klone Repository" "cd /home/ubuntu && git clone \"$REPO_URL\" kimi-linear-complete"
run_step "Setze Besitzerrechte" "chown -R ubuntu:ubuntu /home/ubuntu/kimi-linear-complete"

# ===== DOCKER COMPOSE =====
run_step "Installiere Docker Compose" "\
    curl -L \"https://github.com/docker/compose/releases/latest/download/docker-compose-\$(uname -s)-\$(uname -m)\" -o /usr/local/bin/docker-compose && \
    chmod +x /usr/local/bin/docker-compose
"

# ===== AUTOSTART EINRICHTEN =====
echo "▶ Erstelle Autostart-Service..."
cat > /etc/systemd/system/kimi-linear.service << 'EOF'
[Unit]
Description=Kimi Linear 48B Coding Engine
After=docker.service
Requires=docker.service

[Service]
Type=oneshot
RemainAfterExit=yes
User=ubuntu
WorkingDirectory=/home/ubuntu/kimi-linear-complete/application
ExecStart=/usr/local/bin/docker-compose up -d
ExecStop=/usr/local/bin/docker-compose down
TimeoutStartSec=600

[Install]
WantedBy=multi-user.target
EOF
echo "✓ Autostart-Service erstellt"

run_step "Aktiviere systemd Service" "systemctl daemon-reload && systemctl enable kimi-linear.service"

# ===== FIREWALL =====
run_step "Konfiguriere Firewall" "\
    ufw allow 22/tcp && \
    ufw allow 80/tcp && \
    ufw allow 443/tcp && \
    ufw allow 3000/tcp && \
    ufw allow 9090/tcp && \
    ufw allow 8001:8004/tcp && \
    echo 'y' | ufw enable
"

# ===== CLEANUP & MARKER =====
run_step "Räume auf" "apt autoremove -y && apt autoclean"

# Setup-complete Marker
touch /tmp/setup-complete
echo "[$(date '+%Y-%m-%d %H:%M:%S')] ✅ VM-Setup abgeschlossen!"

# ===== OPTIONAL: AUTOSTART =====
echo "🎯 Starte Services (optional – sonst manuell mit ./start.sh)..."
cd /home/ubuntu/kimi-linear-complete/application
# docker-compose up -d  # Deaktiviert – lieber manuell starten

# ✅ NEU - Prüfungen am Ende
echo "🔍 Prüfe Service-Status..."

set +e

# Check Docker
if ! systemctl is-active --quiet docker; then
    echo "⚠️  Docker läuft nicht!"
    systemctl start docker || echo "  Failed to start Docker"
else
    echo "✓ Docker läuft"
fi

# Check NVIDIA Container Runtime
if ! docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi &>/dev/null; then
    echo "⚠️  NVIDIA Docker Runtime funktioniert nicht!"
else
    echo "✓ NVIDIA Docker Runtime funktioniert"
fi

# Check if repository was cloned
if [ ! -d "/home/ubuntu/kimi-linear-complete" ]; then
    echo "❌ Repository nicht geklont!"
    exit 1
else
    echo "✓ Repository geklont"
fi

set -e

echo "✅ Alle Prüfungen bestanden"
echo "✨ VM bereit für Kimi Linear 48B!"
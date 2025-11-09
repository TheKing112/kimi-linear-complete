#!/bin/bash
# =============================================================================
# KIMI LINEAR - VM INTERNAL SETUP
# Wird automatisch in der VM ausgeführt
# =============================================================================

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

# ===== SYSTEM UPDATE =====
echo "🔧 System-Update..."
apt update && apt upgrade -y
apt install -y curl git build-essential software-properties-common

# ===== NVIDIA DRIVERS =====
echo "🎮 Installiere NVIDIA Treiber..."
apt install -y nvidia-driver-535 nvidia-dkms-535
nvidia-smi || echo "⚠️  GPU noch nicht erkannt – nach Reboot prüfen"

# ===== DOCKER =====
echo "🐳 Installiere Docker..."
curl -fsSL https://get.docker.com | sh
usermod -aG docker ubuntu

# ===== NVIDIA DOCKER =====
echo "🔥 Installiere NVIDIA Docker Runtime..."
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | tee /etc/apt/sources.list.d/nvidia-docker.list
apt update && apt install -y nvidia-docker2
systemctl restart docker

# ===== REPOSITORY KLONEN =====
echo "📥 Klone Repository..."
cd /home/ubuntu
git clone "$REPO_URL" kimi-linear-complete
chown -R ubuntu:ubuntu kimi-linear-complete

# ===== DOCKER COMPOSE =====
echo "🐋 Installiere Docker Compose..."
curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
chmod +x /usr/local/bin/docker-compose

# ===== AUTOSTART EINRICHTEN =====
echo "🚀 Erstelle Autostart-Service..."
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

systemctl daemon-reload
systemctl enable kimi-linear.service

# ===== FIREWALL =====
echo "🔓 Öffne Ports..."
ufw allow 22/tcp
ufw allow 80/tcp
ufw allow 443/tcp
ufw allow 3000/tcp
ufw allow 9090/tcp
ufw allow 8001:8004/tcp
echo "y" | ufw enable

# ===== CLEANUP & MARKER =====
echo "🧹 Räume auf..."
apt autoremove -y
apt autoclean

# Setup-complete Marker
touch /tmp/setup-complete
echo "[$(date '+%Y-%m-%d %H:%M:%S')] ✅ VM-Setup abgeschlossen!"

# ===== OPTIONAL: AUTOSTART =====
echo "🎯 Starte Services (optional – sonst manuell mit ./start.sh)..."
cd /home/ubuntu/kimi-linear-complete/application
# docker-compose up -d  # Deaktiviert – lieber manuell starten

echo "✨ VM bereit für Kimi Linear 48B!"
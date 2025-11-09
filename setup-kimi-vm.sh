#!/bin/bash
# =============================================================================
# KIMI LINEAR 48B - ONE-COMMAND VM SETUP
# Erstellt Google Cloud VM und deployt komplette Coding Engine
# =============================================================================

set -euo pipefail

# ===== CONFIGURATION =====
readonly VM_NAME="kimi-linear-auto"
readonly VM_TYPE="g2-standard-8"
readonly GPU_TYPE="nvidia-l4"
readonly DISK_SIZE="400GB"
readonly ZONES=("us-east1-b" "us-east1-c" "us-central1-a" "us-central1-b" "us-west1-b")
readonly REPO_URL="https://github.com/dein-repo/kimi-linear-complete.git"  # ⚠️ Anpassen!
readonly SETUP_SCRIPT_URL="https://raw.githubusercontent.com/dein-repo/kimi-linear-complete/main/setup-in-vm.sh"

# ===== COLORS =====
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly BLUE='\033[0;34m'
readonly YELLOW='\033[1;33m'
readonly NC='\033[0m'

# ===== LOGGING =====
log() { echo -e "${BLUE}[$(date '+%H:%M:%S')]${NC} ▶ $*"; }
ok() { echo -e "${GREEN}✓${NC} $*"; }
error() { echo -e "${RED}✗${NC} $*" >&2; exit 1; }
warn() { echo -e "${YELLOW}⚠${NC} $*"; }

# ===== VALIDATION =====
check_gcloud() {
    if ! command -v gcloud &>/dev/null; then
        error "gcloud CLI nicht installiert. Bitte installieren: https://cloud.google.com/sdk/docs/install"
    fi
    ok "gcloud authentifiziert: $(gcloud config get-value account 2>/dev/null || echo 'Nicht authentifiziert')"
}

check_git() {
    if ! command -v git &>/dev/null; then
        error "git nicht installiert"
    fi
}

# ===== VM CREATION =====
find_available_zone() {
    log "Suche verfügbare GPU-Kapazität..."
    for zone in "${ZONES[@]}"; do
        if gcloud compute instances create "$VM_NAME" \
            --zone="$zone" \
            --machine-type="$VM_TYPE" \
            --accelerator="type=$GPU_TYPE,count=1" \
            --image-family="ubuntu-2204-lts" \
            --image-project="ubuntu-os-cloud" \
            --boot-disk-size="$DISK_SIZE" \
            --boot-disk-type="pd-ssd" \
            --maintenance-policy="TERMINATE" \
            --scopes="cloud-platform" \
            --tags="ai-workstation" \
            --dry-run &>/dev/null; then
            echo "$zone"
            return 0
        fi
    done
    return 1
}

create_vm() {
    local zone
    zone=$(find_available_zone) || error "Keine GPU-Kapazität in allen Zonen verfügbar"
    
    log "Erstelle VM in Zone: $zone"
    
    gcloud compute instances create "$VM_NAME" \
        --zone="$zone" \
        --machine-type="$VM_TYPE" \
        --accelerator="type=$GPU_TYPE,count=1" \
        --image-family="ubuntu-2204-lts" \
        --image-project="ubuntu-os-cloud" \
        --boot-disk-size="$DISK_SIZE" \
        --boot-disk-type="pd-ssd" \
        --maintenance-policy="TERMINATE" \
        --scopes="cloud-platform" \
        --tags="ai-workstation" \
        --metadata="startup-script-url=$SETUP_SCRIPT_URL" \
        --quiet
    
    ok "VM erstellt: $VM_NAME in $zone"
    echo "$zone"
}

# ===== SSH SETUP =====
setup_ssh_access() {
    local zone="$1"
    local external_ip
    
    log "Warte auf VM-Startup (30s)..."
    sleep 30
    
    external_ip=$(gcloud compute instances describe "$VM_NAME" \
        --zone="$zone" \
        --format="get(networkInterfaces[0].accessConfigs[0].natIP)")
    
    if [[ -z "$external_ip" ]]; then
        error "Konnte keine externe IP ermitteln"
    fi
    
    # SSH-Konfiguration
    gcloud compute config-ssh --quiet
    
    cat >> ~/.ssh/config << EOF

# Kimi Linear VM
Host kimi-vm
    HostName $external_ip
    User ubuntu
    IdentityFile ~/.ssh/google_compute_engine
    StrictHostKeyChecking no
    Port 22
EOF
    
    ok "SSH konfiguriert: ssh kimi-vm"
    ok "Externe IP: $external_ip"
}

# ===== MONITORING =====
wait_for_setup_completion() {
    local zone="$1"
    local max_attempts=60  # 10 Minuten
    local attempt=0
    
    log "Warte auf Setup-Abschluss in VM..."
    
    while [[ $attempt -lt $max_attempts ]]; do
        if gcloud compute ssh "$VM_NAME" --zone="$zone" --command="test -f /tmp/setup-complete" 2>/dev/null; then
            ok "✅ VM-Setup abgeschlossen!"
            return 0
        fi
        
        echo -n "."
        sleep 10
        ((attempt++))
    done
    
    error "Setup-Timeout nach 10 Minuten"
}

# ===== STATUS & ACCESS =====
show_completion_info() {
    local zone="$1"
    
    log "Hole Zugriffsinformationen..."
    
    local external_ip
    external_ip=$(gcloud compute instances describe "$VM_NAME" \
        --zone="$zone" \
        --format="get(networkInterfaces[0].accessConfigs[0].natIP)")
    
    clear
    cat << EOF
╔════════════════════════════════════════════════════════════════════╗
║                                                                    ║
║   🎉 KIMI LINEAR 48B - VM SETUP COMPLETE!                         ║
║                                                                    ║
║   🖥️  VM Name:     $VM_NAME                                        ║
║   🌐 Externe IP:   $external_ip                                    ║
║   🔗 SSH Access:   ssh kimi-vm                                     ║
║                                                                    ║
║   🚀 Services starten:                                             ║
║   ssh kimi-vm                                                      ║
║   cd kimi-linear-complete                                          ║
║   ./start.sh                                                       ║
║                                                                    ║
║   📊 Monitoring:                                                   ║
║   • Grafana:    http://$external_ip:3000                          ║
║   • Prometheus: http://$external_ip:9090                          ║
║   • Kimi API:   http://$external_ip:8003/docs                     ║
║   • Cognee API: http://$external_ip:8001/docs                     ║
║                                                                    ║
║   💡 Discord Bot Token in .env eintragen!                          ║
║                                                                    ║
╚════════════════════════════════════════════════════════════════════╝
EOF
}

# ===== CLEANUP =====
cleanup_existing_vm() {
    if gcloud compute instances describe "$VM_NAME" --zone=us-east1-b &>/dev/null 2>&1; then
        warn "VM existiert bereits"
        read -p "Löschen und neu erstellen? [y/N] " -r
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            log "Lösche bestehende VM..."
            gcloud compute instances delete "$VM_NAME" --zone=us-east1-b --quiet
            ok "Alte VM gelöscht"
        else
            error "Abbruch - VM existiert bereits"
        fi
    fi
}

# ===== MAIN =====
main() {
    clear
    cat << "EOF"
╔════════════════════════════════════════════════════════════════════╗
║                                                                    ║
║   🚀 KIMI LINEAR 48B - ONE-COMMAND SETUP                         ║
║   Automatische VM-Erstellung + Coding Engine Deployment          ║
║                                                                    ║
║   • Google Cloud VM mit GPU                                       ║
║   • Kimi Linear 48B Sparse Activation                             ║
║   • Discord Bot + GitHub Integration                              ║
║   • Monitoring + Cognee Memory                                    ║
║                                                                    ║
╚════════════════════════════════════════════════════════════════════╝
EOF
    
    check_gcloud
    check_git
    cleanup_existing_vm
    
    local zone
    zone=$(create_vm)
    setup_ssh_access "$zone"
    wait_for_setup_completion "$zone"
    show_completion_info "$zone"
    
    log "✨ Fertig! Deine KI-Coding-Engine läuft auf der VM"
}

# ===== SCRIPT START =====
main "$@"
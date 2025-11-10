```bash
#!/bin/bash
# =============================================================================
# KIMI LINEAR 48B - ONE-COMMAND VM SETUP
# Erstellt Google Cloud VM und deployt komplette Coding Engine
# =============================================================================

set -euo pipefail

# ===== COLORS & LOGGING (Zuerst definieren!) =====
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly BLUE='\033[0;34m'
readonly YELLOW='\033[1;33m'
readonly NC='\033[0m'

log() { echo -e "${BLUE}[$(date '+%H:%M:%S')]${NC} ▶ $*"; }
ok() { echo -e "${GREEN}✓${NC} $*"; }
error() { echo -e "${RED}✗${NC} $*" >&2; exit 1; }
warn() { echo -e "${YELLOW}⚠${NC} $*"; }

# ===== TRAP & CLEANUP =====
trap cleanup_on_interrupt INT TERM

cleanup_on_interrupt() {
    warn "\n\n⚠️  Setup wurde abgebrochen!"
    
    read -p "VM löschen? [y/N] " -r
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        log "Lösche VM..."
        gcloud compute instances delete "$VM_NAME" --quiet --zone=us-east1-b 2>/dev/null || true
        ok "VM gelöscht"
    else
        warn "VM wurde erstellt, aber Setup unvollständig"
        warn "Manuell löschen mit: gcloud compute instances delete $VM_NAME"
    fi
    
    exit 130
}

# ===== CONFIGURATION =====
readonly VM_NAME="kimi-linear-auto"
readonly VM_TYPE="g2-standard-8"
readonly GPU_TYPE="nvidia-l4"
readonly DISK_SIZE="400GB"
readonly ZONES=("us-east1-b" "us-east1-c" "us-central1-a" "us-central1-b" "us-west1-b")

# ===== VALIDATION =====
# Repository URL - kann als Parameter oder Umgebungsvariable gesetzt werden
REPO_URL="${1:-${KIMI_REPO_URL:-}}"

if [[ -z "$REPO_URL" ]]; then
    error "Repository URL erforderlich!\nUsage: $0 <github-repo-url>\n   or: export KIMI_REPO_URL=<url>"
fi

# Validiere GitHub URL
if [[ ! "$REPO_URL" =~ ^https://github.com/ ]]; then
    error "Ungültige GitHub URL: '$REPO_URL'. Muss mit https://github.com/ beginnen."
fi

# Leite Setup-Script URL aus Repository URL ab
# Konvertiert https://github.com/user/repo.git -> https://raw.githubusercontent.com/user/repo/main/setup-in-vm.sh
SETUP_SCRIPT_URL="${REPO_URL/github.com/raw.githubusercontent.com}"
SETUP_SCRIPT_URL="${SETUP_SCRIPT_URL%.git}/main/setup-in-vm.sh"

# ===== VALIDATION FUNCTIONS =====
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
    log "Suche verfügbare GPU-Kapazität in ${#ZONES[@]} Zonen..."
    
    local attempt=1
    local max_attempts=2
    
    while [ $attempt -le $max_attempts ]; do
        log "Versuch $attempt von $max_attempts..."
        
        for zone in "${ZONES[@]}"; do
            log "  Prüfe Zone: $zone"
            
            # Check quota first
            if ! gcloud compute project-info describe \
                --format="value(quotas.filter(metric:GPUS))" 2>/dev/null | grep -q "GPUS"; then
                warn "  GPU quota check failed for $zone"
                continue
            fi
            
            # Try dry-run
            if gcloud compute instances create "$VM_NAME-test" \
                --zone="$zone" \
                --machine-type="$VM_TYPE" \
                --accelerator="type=$GPU_TYPE,count=1" \
                --dry-run &>/dev/null; then
                ok "  Verfügbar: $zone"
                echo "$zone"
                return 0
            else
                warn "  Nicht verfügbar: $zone"
            fi
        done
        
        if [ $attempt -lt $max_attempts ]; then
            log "Keine Zone verfügbar, warte 30s vor erneutem Versuch..."
            sleep 30
        fi
        
        ((attempt++))
    done
    
    error "Keine GPU-Kapazität in allen Zonen nach $max_attempts Versuchen"
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

# ===== FIREWALL SETUP =====
setup_firewall_rules() {
    local zone="$1"
    log "Konfiguriere Firewall-Regeln..."
    
    # Check if firewall rules exist
    local rules=("allow-http" "allow-https" "allow-ssh" "allow-custom-ports")
    
    for rule in "${rules[@]}"; do
        if ! gcloud compute firewall-rules describe "kimi-$rule" &>/dev/null; then
            log "Erstelle Firewall-Regel: kimi-$rule"
            
            case "$rule" in
                allow-http)
                    gcloud compute firewall-rules create "kimi-$rule" \
                        --allow=tcp:80 \
                        --target-tags=ai-workstation \
                        --description="Allow HTTP traffic"
                    ;;
                allow-https)
                    gcloud compute firewall-rules create "kimi-$rule" \
                        --allow=tcp:443 \
                        --target-tags=ai-workstation \
                        --description="Allow HTTPS traffic"
                    ;;
                allow-ssh)
                    gcloud compute firewall-rules create "kimi-$rule" \
                        --allow=tcp:22 \
                        --target-tags=ai-workstation \
                        --description="Allow SSH access"
                    ;;
                allow-custom-ports)
                    gcloud compute firewall-rules create "kimi-$rule" \
                        --allow=tcp:3000,tcp:8001-8004,tcp:9090 \
                        --target-tags=ai-workstation \
                        --description="Allow Kimi services"
                    ;;
            esac
        else
            log "Firewall-Regel existiert bereits: kimi-$rule"
        fi
    done
    
    ok "Firewall-Regeln konfiguriert"
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

show_cost_estimate() {
    cat << EOF

╔════════════════════════════════════════════════════════════════════╗
║  💰 GESCHÄTZTE KOSTEN                                              ║
╚════════════════════════════════════════════════════════════════════╝

  • VM (g2-standard-8):     ~\$0.90/Stunde
  • GPU (NVIDIA L4):         ~\$0.75/Stunde
  • Disk (400GB SSD):        ~\$0.17/Tag
  
  📊 Gesamt:                  ~\$1.65/Stunde  oder  ~\$1,200/Monat
  
  ⚠️  Diese Schätzung gilt bei 100% Auslastung!
  
  💡 TIPP: Stoppe VM wenn nicht verwendet:
     gcloud compute instances stop $VM_NAME --zone=<zone>

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
    setup_firewall_rules "$zone"
    setup_ssh_access "$zone"
    wait_for_setup_completion "$zone"
    show_cost_estimate
    show_completion_info "$zone"
    
    log "✨ Fertig! Deine KI-Coding-Engine läuft auf der VM"
}

# ===== SCRIPT START =====
main "$@"
```
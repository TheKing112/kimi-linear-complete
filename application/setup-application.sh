#!/bin/bash
# =============================================================================
# KIMI LINEAR 48B - APPLICATION SETUP (SPARSE ACTIVATION)
# Deploys all services: Discord Bot, Cognee, Kimi Linear, GitHub Integration, Monitoring
# =============================================================================

set -euo pipefail

readonly LOG_FILE="/tmp/application-setup.log"

log() { echo "[$(date +%T)] $1" | tee -a "$LOG_FILE"; }
ok()  { echo "✓ $1" | tee -a "$LOG_FILE"; }
error() { echo "✗ $1" | tee -a "$LOG_FILE"; exit 1; }
warn() { echo "⚠ $1" | tee -a "$LOG_FILE"; }

# ==================================== CONFIGURATION ====================================

readonly APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ==================================== SERVICE DEFINITIONS ====================================

create_directories() {
    log "Creating service directories..."
    
    mkdir -p services/{kimi-linear,cognee,discord-bot,github-integration}
    mkdir -p services/kimi-linear/src
    mkdir -p services/cognee/src
    mkdir -p services/discord-bot/src
    mkdir -p services/github-integration/src
    mkdir -p monitoring/{prometheus,grafana/provisioning/{dashboards,datasources}}
    mkdir -p init-scripts
    mkdir -p logs/{kimi-linear,cognee,discord-bot,github-integration}
    
    ok "Created directory structure"
}

create_env_file() {
    # ✅ Check write permissions
    if [ ! -w . ]; then
        error "No write permission in current directory"
    fi
    
    # ✅ Check disk space
    local available_kb=$(df -k . | tail -1 | awk '{print $4}')
    if [ "$available_kb" -lt 1024 ]; then
        error "Insufficient disk space (< 1MB available)"
    fi
    
    # Use temp file for atomic operations
    local temp_env=$(mktemp) || error "Cannot create temp file"
    
    # Trigger error on failure inside the function
    trap 'rm -f "$temp_env"; error "Failed to create .env file"' ERR
    
    if [[ -f .env ]]; then
        log "Backing up existing .env..."
        backup_name=".env.backup.$(date +%s)"
        cp .env "$backup_name" || error "Failed to create backup"
        ok "Backup created: $backup_name"
        
        # Ask if user wants to overwrite
        read -p "Overwrite existing .env? [y/N] " -r
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            log "Keeping existing .env"
            rm -f "$temp_env"
            trap - ERR
            return 0
        fi
    fi
    
    log "Creating comprehensive .env file..."
    
    # Generate secure passwords
    POSTGRES_PASSWORD=$(openssl rand -base64 32)
    GRAFANA_PASSWORD=$(openssl rand -base64 16)
    GITHUB_WEBHOOK_SECRET=$(openssl rand -hex 32)
    
    cat > "$temp_env" << EOF
# =============================================================================
# KIMI LINEAR 48B - PRODUCTION ENVIRONMENT
# =============================================================================

# Discord Bot (REQUIRED)
DISCORD_BOT_TOKEN=your_discord_token_here
DISCORD_GUILD_ID=

# Kimi Linear 48B Model (CRITICAL: fla-core for A3B)
MODEL_NAME=moonshotai/Kimi-Linear-48B-A3B-Instruct
MODEL_PATH=/models/kimi-linear-48b
DEVICE=cuda
LOAD_IN_4BIT=true
TORCH_COMPILE=true
TRUST_REMOTE_CODE=true
MAX_INPUT_LENGTH=1048576
MAX_BATCH_TOKENS=8192
GPU_COUNT=1
CUDA_VISIBLE_DEVICES=0

# Hugging Face (Optional)
HF_TOKEN=

# Database
POSTGRES_PASSWORD=${POSTGRES_PASSWORD}
POSTGRES_USER=cognee
POSTGRES_DB=cognee
POSTGRES_HOST=postgres
POSTGRES_PORT=5432
DATABASE_URL=postgresql://cognee:${POSTGRES_PASSWORD}@postgres:5432/cognee

# Redis
REDIS_URL=redis://redis:6379/0

# API URLs
KIMI_LINEAR_URL=http://kimi-linear:8003
COGNEE_URL=http://cognee:8001
GITHUB_INTEGRATION_URL=http://github-integration:8004

# GitHub Integration (REQUIRED for autonomous editing)
GITHUB_TOKEN=your_github_token_here
GITHUB_WEBHOOK_SECRET=${GITHUB_WEBHOOK_SECRET}
GITHUB_AUTO_REVIEW=true
GITHUB_AUTO_CREATE_PR=false

# Autonomous Editing (SECURITY)
AUTONOMOUS_MODE_ENABLED=true
AUTONOMOUS_REQUIRE_APPROVAL=true
AUTONOMOUS_MAX_TOKENS=4096
AUTONOMOUS_TEMPERATURE=0.2
AUTONOMOUS_CREATE_BRANCH=true
AUTONOMOUS_BRANCH_PREFIX=autonomous/
AUTONOMOUS_COMMIT_MESSAGE_TEMPLATE="[autonomous] {description} by Kimi Linear"
AUTONOMOUS_REVIEW_TRIGGER="[auto-review]"

# AI APIs (Optional)
ANTHROPIC_API_KEY=
OPENAI_API_KEY=

# Monitoring
GRAFANA_PASSWORD=${GRAFANA_PASSWORD}
GF_SECURITY_ADMIN_PASSWORD=${GRAFANA_PASSWORD}
GF_INSTALL_PLUGINS=grafana-clock-panel,grafana-simple-json-datasource

# Network
NETWORK_NAME=kimi-network

# Logging
LOG_LEVEL=INFO

# Discord Control Variables
DISCORD_CONTROL_USERS=
DISCORD_CONTROL_CHANNEL=
DISCORD_OWNER_ID=
EOF
    
    # Atomic move from temp file to final location
    mv "$temp_env" .env || error "Failed to create .env"
    
    # Set secure permissions
    chmod 600 .env || error "Failed to set .env permissions"
    
    # Clear trap
    trap - ERR
    
    ok "Created .env with all variables (PLEASE EDIT TOKENS!)"
}

validate_env_file() {
    log "Validating .env file..."
    
    required_vars=(
        "DISCORD_BOT_TOKEN"
        "POSTGRES_PASSWORD"
        "MODEL_NAME"
        "NETWORK_NAME"
    )
    
    missing_vars=()
    
    for var in "${required_vars[@]}"; do
        if ! grep -q "^${var}=" .env || grep -q "^${var}=your_" .env; then
            missing_vars+=("$var")
        fi
    done
    
    if [ ${#missing_vars[@]} -gt 0 ]; then
        warn "Missing or unconfigured variables:"
        printf '  - %s\n' "${missing_vars[@]}" | tee -a "$LOG_FILE"
        warn "Please edit .env before starting services"
        return 1
    fi
    
    ok "Environment variables validated"
    return 0
}

create_init_scripts() {
    log "Creating database init scripts..."
    
    # Base schema
    cat > init-scripts/01-init.sql << 'EOF'
-- ============================================================================
-- Cognee Database Schema (Base)
-- ============================================================================

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS memories (
    id SERIAL PRIMARY KEY,
    user_id VARCHAR(255) NOT NULL,
    content TEXT NOT NULL,
    embedding vector(384),
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_memories_user ON memories(user_id);
CREATE INDEX idx_memories_created ON memories(created_at DESC);

CREATE TABLE IF NOT EXISTS knowledge_graph (
    id SERIAL PRIMARY KEY,
    user_id VARCHAR(255),
    subject TEXT,
    predicate TEXT,
    object TEXT,
    confidence FLOAT DEFAULT 0.8,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_knowledge_user ON knowledge_graph(user_id);

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
EOF

    # Extended schema
    cat > init-scripts/02-extended.sql << 'EOF'
-- ============================================================================
-- Extended Memory Schema
-- ============================================================================

CREATE TABLE IF NOT EXISTS generation_metadata (
    memory_id INTEGER PRIMARY KEY REFERENCES memories(id) ON DELETE CASCADE,
    prompt_text TEXT,
    prompt_tokens INTEGER,
    generated_tokens INTEGER,
    total_tokens INTEGER,
    generation_time_ms INTEGER,
    temperature FLOAT,
    top_p FLOAT,
    model_name VARCHAR(255),
    model_version VARCHAR(50),
    gpu_device VARCHAR(50),
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS user_feedback (
    id SERIAL PRIMARY KEY,
    memory_id INTEGER REFERENCES memories(id) ON DELETE CASCADE,
    user_id VARCHAR(255),
    feedback_type VARCHAR(20),
    feedback_data JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS conversation_threads (
    id SERIAL PRIMARY KEY,
    thread_id VARCHAR(255) UNIQUE,
    user_id VARCHAR(255),
    initial_prompt TEXT,
    context JSONB,
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_activity TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

ALTER TABLE memories
ADD COLUMN IF NOT EXISTS embedding_version VARCHAR(20) DEFAULT 'all-MiniLM-L6-v2-v1',
ADD COLUMN IF NOT EXISTS tags TEXT[],
ADD COLUMN IF NOT EXISTS is_autonomous BOOLEAN DEFAULT false,
ADD COLUMN IF NOT EXISTS parent_memory_id INTEGER REFERENCES memories(id);

CREATE INDEX idx_memories_tags ON memories USING GIN(tags);
CREATE INDEX idx_memories_autonomous ON memories(is_autonomous) WHERE is_autonomous = true;
EOF

    ok "Created database init scripts"
}

create_docker_compose() {
    log "Creating docker-compose.yml for Sparse Activation..."

    cat > docker-compose.yml << 'EOF'
version: '3.8'

services:
  # PostgreSQL with vector support
  postgres:
    image: pgvector/pgvector:pg15
    container_name: kimi-postgres-48b
    environment:
      POSTGRES_DB: ${POSTGRES_DB}
      POSTGRES_USER: ${POSTGRES_USER}
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./init-scripts:/docker-entrypoint-initdb.d
    ports:
      - "5432:5432"
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U ${POSTGRES_USER}"]
      interval: 10s
      timeout: 5s
      retries: 5
    networks:
      - ${NETWORK_NAME}
    restart: unless-stopped

  # Redis Cache
  redis:
    image: redis:7-alpine
    container_name: kimi-redis-48b
    volumes:
      - redis_data:/data
    command: redis-server --appendonly yes --maxmemory 2gb --maxmemory-policy allkeys-lru
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 5s
      timeout: 3s
      retries: 5
    networks:
      - ${NETWORK_NAME}
    restart: unless-stopped

  # Kimi Linear 48B Model Service
  kimi-linear:
    build:
      context: ./services/kimi-linear
      dockerfile: Dockerfile
    container_name: kimi-linear-48b-sparse
    shm_size: 16gb
    ipc: host
    environment:
      - MODEL_NAME=${MODEL_NAME}
      - MODEL_PATH=${MODEL_PATH}
      - DEVICE=${DEVICE}
      - LOAD_IN_4BIT=${LOAD_IN_4BIT}
      - TORCH_COMPILE=${TORCH_COMPILE}
      - TRUST_REMOTE_CODE=${TRUST_REMOTE_CODE}
      - MAX_INPUT_LENGTH=${MAX_INPUT_LENGTH}
      - PORT=8003
      - CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}
      - HF_TOKEN=${HF_TOKEN}
    volumes:
      - model_cache:${MODEL_PATH}
      - ./services/kimi-linear:/app
      - ~/.cache/huggingface:/root/.cache/huggingface
    ports:
      - "8003:8003"
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_healthy
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: ${GPU_COUNT}
              capabilities: [gpu]
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8003/health"]
      interval: 60s
      timeout: 30s
      retries: 5
      start_period: 300s
    networks:
      - ${NETWORK_NAME}
    restart: unless-stopped

  # Cognee Memory Service
  cognee:
    build:
      context: ./services/cognee
      dockerfile: Dockerfile
    container_name: kimi-cognee-48b
    environment:
      - DATABASE_URL=${DATABASE_URL}
      - REDIS_URL=${REDIS_URL}
      - OPENAI_API_KEY=${OPENAI_API_KEY}
    volumes:
      - ./services/cognee:/app
    ports:
      - "8001:8001"
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_healthy
    networks:
      - ${NETWORK_NAME}
    restart: unless-stopped

  # Discord Bot
  discord-bot:
    build:
      context: ./services/discord-bot
      dockerfile: Dockerfile
    container_name: kimi-discord-48b
    environment:
      - DISCORD_BOT_TOKEN=${DISCORD_BOT_TOKEN}
      - DISCORD_CONTROL_USERS=${DISCORD_CONTROL_USERS}
      - DISCORD_CONTROL_CHANNEL=${DISCORD_CONTROL_CHANNEL}
      - DISCORD_OWNER_ID=${DISCORD_OWNER_ID}
      - KIMI_LINEAR_URL=${KIMI_LINEAR_URL}
      - COGNEE_URL=${COGNEE_URL}
    volumes:
      - ./services/discord-bot/logs:/app/logs
      - ./models:/models:ro
    depends_on:
      - kimi-linear
      - cognee
    networks:
      - ${NETWORK_NAME}
    restart: unless-stopped

  # GitHub Integration
  github-integration:
    build:
      context: ./services/github-integration
      dockerfile: Dockerfile
    container_name: kimi-github-integration
    environment:
      - GITHUB_TOKEN=${GITHUB_TOKEN}
      - COGNEE_URL=${COGNEE_URL}
      - KIMI_LINEAR_URL=${KIMI_LINEAR_URL}
      - REDIS_URL=${REDIS_URL}
      - AUTONOMOUS_MODE_ENABLED=${AUTONOMOUS_MODE_ENABLED}
      - AUTONOMOUS_REQUIRE_APPROVAL=${AUTONOMOUS_REQUIRE_APPROVAL}
    volumes:
      - ./services/github-integration:/app
      - /tmp/github-clones:/tmp/github-clones
    ports:
      - "8004:8004"
    depends_on:
      - cognee
      - kimi-linear
      - redis
    networks:
      - ${NETWORK_NAME}
    restart: unless-stopped

  # Prometheus Monitoring
  prometheus:
    image: prom/prometheus:latest
    container_name: kimi-prometheus-48b
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    ports:
      - "9090:9090"
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--storage.tsdb.retention.time=30d'
      - '--web.enable-lifecycle'
    networks:
      - ${NETWORK_NAME}
    restart: unless-stopped

  # Grafana Dashboard
  grafana:
    image: grafana/grafana:latest
    container_name: kimi-grafana-48b
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=${GRAFANA_PASSWORD}
      - GF_INSTALL_PLUGINS=${GF_INSTALL_PLUGINS}
    volumes:
      - grafana_data:/var/lib/grafana
      - ./monitoring/grafana/provisioning:/etc/grafana/provisioning
    ports:
      - "3000:3000"
    depends_on:
      - prometheus
    networks:
      - ${NETWORK_NAME}
    restart: unless-stopped

volumes:
  postgres_data:
  redis_data:
  model_cache:
  prometheus_data:
  grafana_data:

networks:
  kimi-network:
    driver: bridge
    name: ${NETWORK_NAME}
EOF
    
    ok "Created docker-compose.yml (Sparse Activation, 1 GPU, GitHub Integration)"
}

create_helper_scripts() {
    log "Creating management scripts..."

    # Start script
    cat > start.sh << 'EOF'
#!/bin/bash
set -e

echo "🚀 Starting Kimi Linear 48B Stack..."
docker-compose up -d --remove-orphans

echo ""
echo "⏳ Waiting for service readiness..."
sleep 10

echo ""
echo "📊 Service Status:"
docker-compose ps --format "table {{.Name}}\t{{.State}}\t{{.Ports}}"

echo ""
echo "🔗 Access Points:"
echo "  • Grafana:        http://localhost:3000"
echo "  • Prometheus:     http://localhost:9090"
echo "  • Kimi API Docs:  http://localhost:8003/docs"
echo "  • Cognee API:     http://localhost:8001/docs"
echo "  • GitHub API:     http://localhost:8004/docs"
echo ""
echo "📋 View logs:   docker-compose logs -f [service-name]"
echo "🛑 Stop:        docker-compose down"
EOF

    # Stop script
    cat > stop.sh << 'EOF'
#!/bin/bash
echo "⏹️  Stopping Kimi Linear 48B..."
docker-compose down --remove-orphans
echo "✅ Stack stopped"
EOF

    # Status script
    cat > status.sh << 'EOF'
#!/bin/bash
echo "════════════════════════════════════════════════════════════"
echo "  KIMI LINEAR 48B - STATUS"
echo "════════════════════════════════════════════════════════════"

echo ""
echo "📦 Container Status:"
docker-compose ps

echo ""
echo "🏥 Health Checks:"
services=("kimi-linear:8003" "cognee:8001" "github-integration:8004" "prometheus:9090" "grafana:3000")

for service in "${services[@]}"; do
    name="${service%%:*}"
    port="${service##*:}"
    
    if curl -sf "http://localhost:$port/health" &>/dev/null || \
       curl -sf "http://localhost:$port/api/health" &>/dev/null; then
        echo "✅ $name is healthy (port $port)"
    else
        echo "⚠  $name not responding (port $port)"
    fi
done

echo ""
echo "📈 GPU Status:"
if command -v nvidia-smi &>/dev/null; then
    nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits
else
    echo "ℹ️  No GPU available (CPU mode)"
fi

echo ""
echo "💾 Disk Usage:"
df -h $(pwd) | tail -1 | awk '{print "Available: " $4 " of " $2 " (" $5 " used)"}'
EOF

    # Logs script
    cat > logs.sh << 'EOF'
#!/bin/bash
if [ $# -eq 0 ]; then
    echo "Usage: ./logs.sh <service-name|all>"
    echo "Services: kimi-linear, cognee, github-integration, discord-bot, postgres, redis, prometheus, grafana"
    exit 1
fi

if [ "$1" == "all" ]; then
    docker-compose logs -f --tail=100
else
    docker-compose logs -f --tail=100 "$1"
fi
EOF

    chmod +x start.sh stop.sh status.sh logs.sh
    
    ok "Created helper scripts"
}

preflight_checks() {
    log "Running preflight checks..."
    
    # ✅ Check write permissions
    if [ ! -w . ]; then
        error "No write permission in current directory"
    fi
    
    # ✅ Check if running as root
    if [ "$EUID" -eq 0 ]; then
        warn "Running as root is not recommended"
        warn "Consider using a non-root user"
    fi
    
    # Check Docker
    if ! command -v docker &>/dev/null; then
        error "Docker not installed"
    fi
    
    # Check Docker Compose (V1 or V2)
    if command -v docker-compose &>/dev/null; then
        log "Docker Compose V1 detected"
    elif docker compose version &>/dev/null 2>&1; then
        log "Docker Compose V2 detected"
    else
        error "Docker Compose not installed"
    fi
    
    # Check disk space
    available_space=$(df -BG . | tail -1 | awk '{print $4}' | sed 's/G//')
    if [ "$available_space" -lt 100 ]; then
        warn "Low disk space: ${available_space}GB available"
        warn "Recommended: 100GB+ for model storage"
    else
        log "Disk space check: ${available_space}GB available"
    fi
    
    # Check nvidia-smi (optional)
    if command -v nvidia-smi &>/dev/null; then
        ok "GPU detected: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
    else
        warn "No GPU detected - CPU mode only"
    fi
    
    ok "Preflight checks complete"
}

check_dependencies() {
    log "Checking dependencies..."
    
    local missing_deps=()
    
    # Check Docker Compose (V1 or V2)
    if ! command -v docker-compose &>/dev/null && \
       ! docker compose version &>/dev/null 2>&1; then
        missing_deps+=("docker-compose")
    fi
    
    # Check openssl for password generation
    if ! command -v openssl &>/dev/null; then
        missing_deps+=("openssl")
    fi
    
    # Check curl for health checks
    if ! command -v curl &>/dev/null; then
        missing_deps+=("curl")
    fi
    
    if [ ${#missing_deps[@]} -gt 0 ]; then
        error "Missing dependencies: ${missing_deps[*]}"
    fi
    
    ok "All dependencies present"
}

validate_setup() {
    log "Validating setup..."
    
    # Check if all required files exist
    local required_files=(
        ".env"
        "docker-compose.yml"
        "start.sh"
        "stop.sh"
        "status.sh"
        "init-scripts/01-init.sql"
    )
    
    for file in "${required_files[@]}"; do
        if [ ! -f "$file" ]; then
            error "Missing file: $file"
        fi
    done
    
    # Check if scripts are executable
    local scripts=("start.sh" "stop.sh" "status.sh" "logs.sh")
    for script in "${scripts[@]}"; do
        if [ ! -x "$script" ]; then
            error "Script not executable: $script"
        fi
    done
    
    # Check .env syntax
    if ! grep -q "^DISCORD_BOT_TOKEN=" .env; then
        error ".env missing required variable: DISCORD_BOT_TOKEN"
    fi
    
    ok "Setup validation passed"
}

# ==================================== MAIN ====================================

main() {
    cat << "EOF"
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║   KIMI LINEAR 48B - APPLICATION SETUP                                       ║
║   Sparse Activation: 48B Total, 3B Active per Forward Pass                   ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
EOF
    
    check_dependencies
    preflight_checks
    
    create_directories
    create_env_file
    validate_env_file || true  # Non-fatal validation
    create_init_scripts
    create_docker_compose
    create_helper_scripts
    
    validate_setup
    
    log ""
    ok "╔══════════════════════════════════════════════════════════════════════════════╗"
    ok "║  SETUP COMPLETE! 🎉                                                          ║"
    ok "║                                                                              ║"
    ok "║  NEXT STEPS:                                                                 ║"
    ok "║  1. Edit .env and add your tokens (CRITICAL!)                               ║"
    ok "║  2. Run: ./start.sh                                                         ║"
    ok "║  3. Wait 5-10 min for model loading                                         ║"
    ok "║  4. Check logs: ./logs.sh kimi-linear                                       ║"
    ok "║                                                                              ║"
    ok "║  ACCESS POINTS:                                                              ║"
    ok "║  • Grafana:        http://localhost:3000                                    ║"
    ok "║  • Prometheus:     http://localhost:9090                                    ║"
    ok "║  • Kimi API Docs:  http://localhost:8003/docs                               ║"
    ok "║  • Cognee API:     http://localhost:8001/docs                               ║"
    ok "║  • GitHub API:     http://localhost:8004/docs                               ║"
    ok "╚══════════════════════════════════════════════════════════════════════════════╝"
}

main "$@" 2>&1 | tee "$LOG_FILE"
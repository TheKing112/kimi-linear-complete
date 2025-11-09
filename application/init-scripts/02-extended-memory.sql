-- =============================================================================
-- ERWEITERTES COGNEE MEMORY SCHEMA
-- Speichert mehr Metadaten für besseren Kontext
-- =============================================================================

-- Table: generation_metadata (NEU)
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

CREATE INDEX idx_generation_meta_model ON generation_metadata(model_name);
CREATE INDEX idx_generation_meta_timestamp ON generation_metadata(timestamp DESC);

-- Table: user_feedback (NEU)
CREATE TABLE IF NOT EXISTS user_feedback (
    id SERIAL PRIMARY KEY,
    memory_id INTEGER REFERENCES memories(id) ON DELETE CASCADE,
    user_id VARCHAR(255),
    feedback_type VARCHAR(20), -- 'thumbs_up', 'thumbs_down', 'edited', 'used'
    feedback_data JSONB, -- Z.B. {"reason": "correct", "edit_diff": "..."}
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_feedback_memory ON user_feedback(memory_id);
CREATE INDEX idx_feedback_user ON user_feedback(user_id);
CREATE INDEX idx_feedback_type ON user_feedback(feedback_type);

-- Table: conversation_threads (NEU)
CREATE TABLE IF NOT EXISTS conversation_threads (
    id SERIAL PRIMARY KEY,
    thread_id VARCHAR(255) UNIQUE, -- Discord-Message-ID oder Session-ID
    user_id VARCHAR(255),
    initial_prompt TEXT,
    context JSONB, -- Chat-Verlauf als Array
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_activity TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_threads_user ON conversation_threads(user_id);
CREATE INDEX idx_threads_active ON conversation_threads(is_active) WHERE is_active = true;

-- Table: test_results (NEU)
CREATE TABLE IF NOT EXISTS test_results (
    id SERIAL PRIMARY KEY,
    memory_id INTEGER REFERENCES generation_metadata(memory_id) ON DELETE CASCADE,
    test_status VARCHAR(20), -- 'passed', 'failed', 'error'
    test_output TEXT,
    test_duration_ms INTEGER,
    coverage_percent FLOAT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_test_status ON test_results(test_status);
CREATE INDEX idx_test_memory ON test_results(memory_id);

-- Table: system_metrics (NEU)
CREATE TABLE IF NOT EXISTS system_metrics (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    gpu_utilization_percent FLOAT,
    gpu_memory_used_mb FLOAT,
    gpu_memory_total_mb FLOAT,
    cpu_utilization_percent FLOAT,
    ram_used_mb FLOAT,
    generation_id INTEGER REFERENCES generation_metadata(memory_id)
);

CREATE INDEX idx_metrics_timestamp ON system_metrics(timestamp DESC);
CREATE INDEX idx_metrics_generation ON system_metrics(generation_id);

-- Erweitere bestehende memories Tabelle mit zusätzlichen Spalten
ALTER TABLE memories
ADD COLUMN IF NOT EXISTS embedding_version VARCHAR(20) DEFAULT 'all-MiniLM-L6-v2-v1',
ADD COLUMN IF NOT EXISTS tags TEXT[], -- Array für Tags: ['bugfix', 'feature', 'refactor']
ADD COLUMN IF NOT EXISTS is_autonomous BOOLEAN DEFAULT false,
ADD COLUMN IF NOT EXISTS parent_memory_id INTEGER REFERENCES memories(id); -- Für Follow-up-Generierungen

-- GIN-Index für Tags (schnelle Suche)
CREATE INDEX IF NOT EXISTS idx_memories_tags ON memories USING GIN(tags);

-- Index für autonome Actions
CREATE INDEX IF NOT EXISTS idx_memories_autonomous ON memories(is_autonomous) WHERE is_autonomous = true;
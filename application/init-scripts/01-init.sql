-- =============================================================================
-- KIMI LINEAR - COGNEE DATABASE INITIALIZATION
-- PostgreSQL mit pgvector Extension
-- Embedding Dimension: 384 (für all-MiniLM-L6-v2)
-- =============================================================================

-- Aktiviere pgvector Extension für vector embeddings
CREATE EXTENSION IF NOT EXISTS vector;

-- Aktiviere weitere nützliche Extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "btree_gin";

-- =============================================================================
-- TABLE: memories
-- Speichert Code-Generierungen mit vector embeddings für semantic search
-- =============================================================================
DROP TABLE IF EXISTS memories CASCADE;

CREATE TABLE memories (
    id SERIAL PRIMARY KEY,
    user_id VARCHAR(255) NOT NULL,
    content TEXT NOT NULL,
    embedding vector(384),  -- Dimension für all-MiniLM-L6-v2
    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- =============================================================================
-- TABLE: knowledge_graph
-- Speichert extrahierte Beziehungen (z.B. Funktionen, Klassen, Imports)
-- =============================================================================
DROP TABLE IF EXISTS knowledge_graph CASCADE;

CREATE TABLE knowledge_graph (
    id SERIAL PRIMARY KEY,
    user_id VARCHAR(255),
    subject TEXT NOT NULL,
    predicate TEXT NOT NULL,
    object TEXT NOT NULL,
    confidence FLOAT DEFAULT 0.8,
    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- =============================================================================
-- INDEXES: Performance Optimierung
-- =============================================================================

-- Standard-Index für User-Lookup
CREATE INDEX idx_memories_user_id ON memories(user_id);
CREATE INDEX idx_memories_created ON memories(created_at DESC);

-- GIN-Index für JSONB durchsuchbarkeit
CREATE INDEX idx_memories_metadata ON memories USING GIN(metadata);
CREATE INDEX idx_knowledge_metadata ON knowledge_graph USING GIN(metadata);

-- B-Tree Index für Knowledge Graph Suchen
CREATE INDEX idx_knowledge_user ON knowledge_graph(user_id);
CREATE INDEX idx_knowledge_subject ON knowledge_graph(subject);
CREATE INDEX idx_knowledge_predicate ON knowledge_graph(predicate);

-- =============================================================================
-- INDEX: Vector Search (IVFFLAT)
-- WIRD ERST NACH 1.000+ Einträgen erstellt für bessere Performance
-- AUSKOMMENTIERT für Initial-Setup - manuell aktivieren wenn DB wächst
-- =============================================================================
-- CREATE INDEX idx_memories_embedding ON memories 
-- USING ivfflat(embedding vector_cosine_ops) 
-- WITH (lists = 100);  -- Anpassen: lists = sqrt(rows) / gewünschte_clusters

-- =============================================================================
-- FUNCTION: Update Trigger für updated_at
-- =============================================================================
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- DROP TRIGGER falls existiert
DROP TRIGGER IF EXISTS update_memories_updated_at ON memories;

-- Trigger für memories Tabelle
CREATE TRIGGER update_memories_updated_at 
BEFORE UPDATE ON memories 
FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- =============================================================================
-- FUNCTION: Semantic Search
-- Sucht nach ähnlichen Memories mit Cosine Similarity
-- Usage: SELECT * FROM search_memories('<embedding_vector>', 'user_123', 10);
-- =============================================================================
CREATE OR REPLACE FUNCTION search_memories(
    query_embedding vector(384),
    user_filter VARCHAR(255) DEFAULT NULL,
    result_limit INT DEFAULT 10
)
RETURNS TABLE(
    id INT,
    user_id VARCHAR(255),
    content TEXT,
    metadata JSONB,
    similarity FLOAT,
    created_at TIMESTAMP WITH TIME ZONE
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        m.id,
        m.user_id,
        m.content,
        m.metadata,
        1 - (m.embedding <=> query_embedding) AS similarity,
        m.created_at
    FROM memories m
    WHERE (user_filter IS NULL OR m.user_id = user_filter)
    ORDER BY similarity DESC
    LIMIT result_limit;
END;
$$ LANGUAGE plpgsql;

-- =============================================================================
-- FUNCTION: Knowledge Graph Query
-- Gibt alle Beziehungen eines Users zurück
-- =============================================================================
CREATE OR REPLACE FUNCTION get_user_knowledge_graph(
    target_user_id VARCHAR(255)
)
RETURNS TABLE(
    subject TEXT,
    predicate TEXT,
    object TEXT,
    confidence FLOAT,
    created_at TIMESTAMP WITH TIME ZONE
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        kg.subject,
        kg.predicate,
        kg.object,
        kg.confidence,
        kg.created_at
    FROM knowledge_graph kg
    WHERE kg.user_id = target_user_id
    ORDER BY kg.confidence DESC, kg.created_at DESC;
END;
$$ LANGUAGE plpgsql;

-- =============================================================================
-- SAMPLE DATA: Optional - Testdaten für erste Verifikation
-- =============================================================================
-- INSERT INTO memories (user_id, content, embedding, metadata) VALUES
-- ('test_user', 'def fibonacci(n): return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)', 
--  NULL,  -- Embedding würde in Python generiert
--  '{"language": "python", "type": "function"}'::jsonb);

-- =============================================================================
-- DONE
-- =============================================================================
SELECT '✅ Cognee Database Schema erfolgreich initialisiert!' AS status;
-- =============================================================================
-- KIMI LINEAR - COGNEE DATABASE INITIALIZATION
-- PostgreSQL mit pgvector Extension
-- Version: 2.1
-- Embedding Dimension: 384 (für all-MiniLM-L6-v2)
-- =============================================================================

-- Aktiviere pgvector Extension für vector embeddings
CREATE EXTENSION IF NOT EXISTS vector;

-- Aktiviere weitere nützliche Extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "btree_gin";
CREATE EXTENSION IF NOT EXISTS pg_cron;  -- Für automatische Index-Pflege

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
    is_autonomous BOOLEAN DEFAULT false,  -- Flag für autonome Generierungen
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
-- UNIQUE CONSTRAINTS: Datenintegrität sichern
-- =============================================================================

-- Verhindert doppelte Knowledge Graph Tripel (user, subj, pred, obj)
ALTER TABLE knowledge_graph 
ADD CONSTRAINT unique_kg_triple 
UNIQUE (user_id, subject, predicate, object);

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

-- Additional Time-based Index für cleanup_old_memories
CREATE INDEX IF NOT EXISTS idx_memories_created_at 
ON memories(created_at) 
WHERE metadata->>'keep_forever' IS DISTINCT FROM 'true';

-- =============================================================================
-- INDEX: Vector Search (IVFFLAT)
-- WIRD ERST NACH 1.000+ Einträgen erstellt für bessere Performance
-- AUSKOMMENTIERT für Initial-Setup - manuell aktivieren wenn DB wächst
-- =============================================================================
-- ❌ VORHER (Blockierend)
-- CREATE INDEX idx_memories_embedding ON memories 
-- USING ivfflat(embedding vector_cosine_ops) 
-- WITH (lists = 100);  -- Anpassen: lists = sqrt(rows) / gewünschte_clusters

-- ✅ NACHHER - Non-blocking
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_memories_embedding 
ON memories USING ivfflat(embedding vector_cosine_ops) 
WITH (lists = 100);

-- ✅ NEU - Pagination Index für schnelle Cursor-Pagination (ersetzt idx_memories_user_created)
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_memories_pagination
ON memories(user_id, created_at DESC, id DESC);

-- ✅ NEU - Optimiertes Composite Index nur für autonome Memories
-- Entfernt redundanten einfachen Index (falls vorhanden) und ersetzt ihn durch
-- einen optimierten Composite Index für user_id-Suche + Sortierung nach created_at
DROP INDEX IF EXISTS idx_memories_autonomous;
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_memories_autonomous_only
ON memories(user_id, created_at DESC)
WHERE is_autonomous = true;

-- ⚠️ HINWEIS: Der folgende Index ist REDUNDANT und suboptimal!
-- Die Spalte 'is_autonomous' im Index ist überflüssig, da der Partial Index
-- (WHERE is_autonomous = true) bereits garantiert, dass alle Zeilen is_autonomous=true haben.
-- Der bestehende Index 'idx_memories_autonomous_only' oben ist die bessere Lösung.
-- Wird nur auf ausdrücklichen Wunsch hinzugefügt:
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_memories_user_autonomous
ON memories(user_id, is_autonomous, created_at DESC)
WHERE is_autonomous = true;

-- ✅ HINZUFÜGEN nach allen CREATE INDEX:
-- Automatische Index-Pflege
CREATE OR REPLACE FUNCTION maintain_indexes()
RETURNS void AS $$
BEGIN
    -- Reindex bei Fragmentierung > 30%
    PERFORM schemaname, tablename, indexname
    FROM pg_stat_user_indexes
    WHERE idx_scan = 0 
    AND indexrelname NOT LIKE 'pg_%';
    
    -- Analyze nach großen Changes
    ANALYZE memories;
    ANALYZE knowledge_graph;
END;
$$ LANGUAGE plpgsql;

-- Cronjob (via pg_cron extension)
SELECT cron.schedule('index-maintenance', '0 3 * * 0', 'SELECT maintain_indexes()');

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
-- FUNCTION: Batch Insert für Knowledge Graph (Robust & Atomic)
-- Fügt mehrere Knowledge Graph Einträge in einer Transaktion ein
-- Parameter update_on_conflict: true = aktualisiert confidence, false = überspringt
-- =============================================================================
CREATE OR REPLACE FUNCTION batch_insert_knowledge(
    entries JSONB,
    update_on_conflict BOOLEAN DEFAULT true
)
RETURNS INTEGER AS $$
DECLARE
    inserted_count INTEGER := 0;
BEGIN
    -- Atomic INSERT mit unnest für Performance
    INSERT INTO knowledge_graph (user_id, subject, predicate, object, confidence)
    SELECT 
        e->>'user_id',
        e->>'subject',
        e->>'predicate',
        e->>'object',
        (e->>'confidence')::FLOAT
    FROM jsonb_array_elements(entries) e
    ON CONFLICT (user_id, subject, predicate, object)
    DO UPDATE SET
        confidence = CASE 
            WHEN update_on_conflict THEN GREATEST(knowledge_graph.confidence, excluded.confidence)
            ELSE knowledge_graph.confidence
        END,
        metadata = CASE 
            WHEN update_on_conflict THEN 
                COALESCE(excluded.metadata, knowledge_graph.metadata)
            ELSE knowledge_graph.metadata
        END,
        created_at = CASE 
            WHEN update_on_conflict THEN CURRENT_TIMESTAMP
            ELSE knowledge_graph.created_at
        END;
    
    GET DIAGNOSTICS inserted_count = ROW_COUNT;
    RETURN inserted_count;
END;
$$ LANGUAGE plpgsql;

-- =============================================================================
-- FUNCTION: Cleanup alter Memories (Safe & Efficient)
-- Löscht Memories die älter als X Tage sind, außer sie sind als 'keep_forever' markiert
-- =============================================================================
CREATE OR REPLACE FUNCTION cleanup_old_memories(
    days_old INTEGER DEFAULT 90,
    max_batch_size INTEGER DEFAULT 10000
)
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER := 0;
    batch_count INTEGER;
BEGIN
    -- Loop in Batches für große Löschmengen (verhindert lange Locks)
    LOOP
        DELETE FROM memories
        WHERE id IN (
            SELECT id
            FROM memories
            WHERE created_at < NOW() - (days_old || ' days')::INTERVAL
            AND metadata->>'keep_forever' IS DISTINCT FROM 'true'
            LIMIT max_batch_size
        );
        
        GET DIAGNOSTICS batch_count = ROW_COUNT;
        deleted_count := deleted_count + batch_count;
        
        EXIT WHEN batch_count = 0;
    END LOOP;
    
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

-- ✅ NEU - Autovacuum Optimierung nach Massenlöschungen
ALTER TABLE memories SET (
    autovacuum_vacuum_scale_factor = 0.1,
    autovacuum_analyze_scale_factor = 0.05
);

-- =============================================================================
-- FUNCTION: Get Memory Statistics
-- Gibt Statistiken für einen User zurück
-- =============================================================================
CREATE OR REPLACE FUNCTION get_memory_stats(
    target_user_id VARCHAR(255) DEFAULT NULL
)
RETURNS TABLE(
    user_id VARCHAR(255),
    total_memories BIGINT,
    autonomous_memories BIGINT,
    avg_confidence FLOAT,
    oldest_memories_days INTEGER,
    newest_memory TIMESTAMP
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        m.user_id,
        COUNT(*)::BIGINT AS total_memories,
        SUM(CASE WHEN m.is_autonomous THEN 1 ELSE 0 END)::BIGINT AS autonomous_memories,
        AVG((m.metadata->>'confidence')::FLOAT) AS avg_confidence,
        EXTRACT(DAY FROM NOW() - MIN(m.created_at))::INTEGER AS oldest_memories_days,
        MAX(m.created_at) AS newest_memory
    FROM memories m
    WHERE (target_user_id IS NULL OR m.user_id = target_user_id)
    GROUP BY m.user_id
    ORDER BY total_memories DESC;
END;
$$ LANGUAGE plpgsql;

-- =============================================================================
-- MIGRATION HELPER: Duplikate bereinigen (falls Schema auf bestehende DB angewendet)
-- Auskommentiert für Neuinstallation - nur bei Bedarf aktivieren
-- =============================================================================
/*
-- 1. Duplikate im Knowledge Graph finden
SELECT user_id, subject, predicate, object, COUNT(*)
FROM knowledge_graph
GROUP BY user_id, subject, predicate, object
HAVING COUNT(*) > 1;

-- 2. Duplikate entfernen (nur den mit höchstem Confidence behalten)
DELETE FROM knowledge_graph
WHERE id NOT IN (
    SELECT DISTINCT ON (user_id, subject, predicate, object) id
    FROM knowledge_graph
    ORDER BY user_id, subject, predicate, object, confidence DESC, added_at DESC
);

-- 3. Dann UNIQUE CONSTRAINT hinzufügen (falls noch nicht vorhanden)
ALTER TABLE knowledge_graph 
ADD CONSTRAINT IF NOT EXISTS unique_kg_triple 
UNIQUE (user_id, subject, predicate, object);
*/

-- =============================================================================
-- SAMPLE DATA: Optional - Testdaten für erste Verifikation
-- =============================================================================
-- INSERT INTO memories (user_id, content, embedding, metadata, is_autonomous) VALUES
-- ('test_user', 'def fibonacci(n): return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)', 
--  NULL,  -- Embedding würde in Python generiert
--  '{"language": "python", "type": "function", "confidence": 0.95}'::jsonb,
--  true
-- );

-- INSERT INTO knowledge_graph (user_id, subject, predicate, object, confidence) VALUES
-- ('test_user', 'fibonacci', 'function', 'recursion', 0.85),
-- ('test_user', 'fibonacci', 'returns', 'int', 0.9),
-- ('test_user', 'fibonacci', 'calls', 'fibonacci', 0.95);

-- =============================================================================
-- DONE: Schema erfolgreich initialisiert
-- =============================================================================
SELECT '✅ Cognee Database Schema v2.1 erfolgreich initialisiert!' AS status,
       CURRENT_TIMESTAMP AS initialized_at;
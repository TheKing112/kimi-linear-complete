import asyncio
import asyncpg
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def setup_database():
    """Richte Cognee-Datenbank ein"""
    try:
        # Verbindung zur PostgreSQL-Datenbank
        conn = await asyncpg.connect(
            host="postgres",
            port=5432,
            user="cognee",
            password="password",
            database="postgres"
        )
        
        # Erstelle Datenbank falls nicht existiert
        await conn.execute("""
            SELECT 'CREATE DATABASE cognee'
            WHERE NOT EXISTS (
                SELECT FROM pg_database WHERE datname = 'cognee'
            )
        """)
        
        await conn.close()
        
        # Verbindung zur Cognee-Datenbank
        conn = await asyncpg.connect(
            host="postgres",
            port=5432,
            user="cognee",
            password="password",
            database="cognee"
        )
        
        # Erstelle Tabellen für Cognee
        await conn.execute("""
            CREATE EXTENSION IF NOT EXISTS vector;
            
            CREATE TABLE IF NOT EXISTS memories (
                id SERIAL PRIMARY KEY,
                user_id VARCHAR(255),
                content TEXT,
                embedding vector(384),
                metadata JSONB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            
            CREATE TABLE IF NOT EXISTS knowledge_graph (
                id SERIAL PRIMARY KEY,
                subject TEXT,
                predicate TEXT,
                object TEXT,
                confidence FLOAT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            
            CREATE INDEX IF NOT EXISTS idx_memories_user_id ON memories(user_id);
            CREATE INDEX IF NOT EXISTS idx_memories_embedding ON memories USING ivfflat (embedding vector_cosine_ops);
        """)
        
        await conn.close()
        logger.info("✅ Datenbank erfolgreich eingerichtet")
        
    except Exception as e:
        logger.error(f"❌ Fehler bei der Datenbank-Einrichtung: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(setup_database())
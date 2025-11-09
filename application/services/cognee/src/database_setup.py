import asyncio
import asyncpg
import logging
import os  # ✅ Für Umgebungsvariablen

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def setup_database():
    """Richte Cognee-Datenbank ein"""
    # Umgebungsvariablen mit sinnvollen Defaults
    POSTGRES_HOST = os.getenv("POSTGRES_HOST", "postgres")
    POSTGRES_PORT = int(os.getenv("POSTGRES_PORT", "5432"))
    POSTGRES_USER = os.getenv("POSTGRES_USER", "cognee")
    POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD")  # ✅ Aus ENV
    POSTGRES_DB = os.getenv("POSTGRES_DB", "cognee")

    # Prüfe, dass Passwort gesetzt ist
    if not POSTGRES_PASSWORD:
        raise ValueError("POSTGRES_PASSWORD Umgebungsvariable ist nicht gesetzt")

    try:
        logger.info(f"🔗 Verbinde mit PostgreSQL auf {POSTGRES_HOST}:{POSTGRES_PORT}...")
        
        # Verbindung zur PostgreSQL Standard-Datenbank für Setup
        conn = await asyncpg.connect(
            host=POSTGRES_HOST,
            port=POSTGRES_PORT,
            user=POSTGRES_USER,
            password=POSTGRES_PASSWORD,
            database="postgres"  # Verwende Standard-Datenbank für DB-Creation
        )
        
        # Prüfe ob Zieldatenbank existiert
        db_exists = await conn.fetchval(
            "SELECT 1 FROM pg_database WHERE datname = $1",
            POSTGRES_DB
        )
        
        if not db_exists:
            logger.info(f"🔄 Datenbank '{POSTGRES_DB}' existiert nicht, erstelle sie...")
            # CREATE DATABASE kann nicht in Transaktion laufen
            # Wir verwenden einen einfachen f-String, da db_name aus ENV kommt
            await conn.execute(f'CREATE DATABASE "{POSTGRES_DB}"')
            logger.info(f"✅ Datenbank '{POSTGRES_DB}' erstellt")
        else:
            logger.info(f"✅ Datenbank '{POSTGRES_DB}' existiert bereits")
        
        await conn.close()
        
        logger.info(f"🔗 Verbinde mit Datenbank '{POSTGRES_DB}'...")
        # Verbindung zur Cognee-Datenbank
        conn = await asyncpg.connect(
            host=POSTGRES_HOST,
            port=POSTGRES_PORT,
            user=POSTGRES_USER,
            password=POSTGRES_PASSWORD,
            database=POSTGRES_DB
        )
        
        logger.info("📦 Erstelle Tabellen und Indizes...")
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
        logger.info(f"✅ Datenbank '{POSTGRES_DB}' erfolgreich eingerichtet")
        
    except Exception as e:
        logger.error(f"❌ Fehler bei der Datenbank-Einrichtung: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(setup_database())
import asyncio
import asyncpg
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_db_config():
    """Holt und validiert die Datenbank-Konfiguration aus Umgebungsvariablen."""
    password = os.getenv("POSTGRES_PASSWORD")
    if not password:
        raise ValueError("❌ POSTGRES_PASSWORD Umgebungsvariable ist nicht gesetzt")
    
    return {
        "host": os.getenv("POSTGRES_HOST", "postgres"),
        "port": int(os.getenv("POSTGRES_PORT", "5432")),
        "user": os.getenv("POSTGRES_USER", "cognee"),
        "password": password,
        "database": os.getenv("POSTGRES_DB", "cognee")
    }

async def setup_database():
    """Richte Cognee-Datenbank ein"""
    config = get_db_config()
    
    try:
        logger.info(f"🔗 Verbinde mit PostgreSQL auf {config['host']}:{config['port']}...")
        
        # Sicherere Datenbank-Erstellung mit Validierung
        async with asyncpg.connect(
            host=config["host"],
            port=config["port"],
            user=config["user"],
            password=config["password"],
            database="postgres"
        ) as conn:
            
            db_exists = await conn.fetchval(
                "SELECT 1 FROM pg_database WHERE datname = $1",
                config["database"]
            )
            
            if not db_exists:
                logger.info(f"🔄 Erstelle Datenbank '{config['database']}'...")
                # Sicherer Datenbankname (nur alphanumerisch, _, - erlaubt)
                if not config["database"].replace("_", "").replace("-", "").isalnum():
                    raise ValueError(f"Ungültiger Datenbankname: {config['database']}")
                
                await conn.execute(f'CREATE DATABASE "{config['database']}"')
                logger.info(f"✅ Datenbank erstellt")
            else:
                logger.info(f"✅ Datenbank existiert bereits")
        
        logger.info(f"🔗 Verbinde mit Datenbank '{config['database']}'...")
        async with asyncpg.connect(
            host=config["host"],
            port=config["port"],
            user=config["user"],
            password=config["password"],
            database=config["database"]
        ) as conn:
            
            logger.info("📦 Erstelle Tabellen und Indizes...")
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
            
        logger.info(f"✅ Datenbank '{config['database']}' erfolgreich eingerichtet")
        
    except Exception as e:
        logger.error(f"❌ Fehler bei der Datenbank-Einrichtung: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(setup_database())
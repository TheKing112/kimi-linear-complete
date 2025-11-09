"""
Cognee Memory API - Vektor-Datenbank für Code-Memories
- Semantic Search mit Sentence-Transformers
- Knowledge Graph Extraktion
- PostgreSQL mit pgvector
"""

import os
import logging
from typing import List, Optional, Dict, Any
from datetime import datetime, timezone

import asyncpg
import numpy as np
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel, Field, validator
from sentence_transformers import SentenceTransformer
import re
import networkx as nx
from asyncpg.exceptions import UniqueViolationError
from fastapi.middleware.cors import CORSMiddleware
import aiofiles

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("cognee")

app = FastAPI(
    title="Cognee Memory API",
    description="Vector memory storage for AI-generated code",
    version="2.0.0"
)

# ✅ NEU - CORS Middleware hinzufügen
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("ALLOWED_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic Models
class MemoryRequest(BaseModel):
    user_id: str = Field(..., min_length=1, max_length=255)
    content: str = Field(..., min_length=10, max_length=100000)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    @validator('content')
    def validate_content(cls, v):
        if len(v.strip()) < 10:
            raise ValueError('Content too short after stripping')
        return v.strip()

class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    user_id: Optional[str] = None
    limit: int = Field(10, ge=1, le=100)

class MemoryResponse(BaseModel):
    id: int
    content: str
    metadata: Dict[str, Any]
    similarity: float
    created_at: datetime

class UserMemoriesResponse(BaseModel):
    user_id: str
    memories: List[Dict[str, Any]]

# Global state
embeddings_model: Optional[SentenceTransformer] = None
db_pool: Optional[asyncpg.Pool] = None

# Konstanten
EMBEDDING_DIM = 384  # Für all-MiniLM-L6-v2
KNOWLEDGE_CONFIDENCE_THRESHOLD = 0.7

# Hilfsfunktionen
def load_embeddings_model():
    """Lade Sentence-Transformer Modell"""
    global embeddings_model
    
    model_name = os.getenv("EMBEDDINGS_MODEL", "all-MiniLM-L6-v2")
    logger.info(f"Lade Embeddings-Modell: {model_name}")
    
    embeddings_model = SentenceTransformer(model_name)
    logger.info("Embeddings-Modell geladen")

def generate_embedding(text: str) -> List[float]:
    """Generiere Embedding für Text"""
    if embeddings_model is None:
        raise RuntimeError("Embeddings-Modell nicht geladen")
    
    embedding = embeddings_model.encode(text, convert_to_tensor=False)
    return embedding.tolist()

async def init_database():
    """Initialisiere Datenbank und erstelle Tabellen"""
    global db_pool
    
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        raise ValueError("DATABASE_URL nicht gesetzt")
    
    # ✅ RICHTIG - Skalierbarer Connection Pool
    db_pool = await asyncpg.create_pool(
        db_url,
        min_size=5,
        max_size=50,
        command_timeout=60,
        max_queries=50000,
        max_inactive_connection_lifetime=300
    )
    
    # Erstelle Tabellen
    async with db_pool.acquire() as conn:
        # Aktiviere pgvector
        await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        
        # Memories Tabelle
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id SERIAL PRIMARY KEY,
                user_id VARCHAR(255) NOT NULL,
                content TEXT NOT NULL,
                embedding vector($1),
                metadata JSONB DEFAULT '{}',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """, EMBEDDING_DIM)
        
        # Knowledge Graph Tabelle
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS knowledge_graph (
                id SERIAL PRIMARY KEY,
                user_id VARCHAR(255),
                subject TEXT NOT NULL,
                predicate TEXT NOT NULL,
                object TEXT NOT NULL,
                confidence FLOAT DEFAULT 0.8,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Indizes
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_memories_user_id ON memories(user_id)")
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_memories_created ON memories(created_at DESC)")
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_knowledge_user ON knowledge_graph(user_id)")
        
        logger.info("Datenbank-Tabellen erstellt")

# ✅ NEU - Transaction-aware Version
async def extract_knowledge_transactional(
    conn: asyncpg.Connection,
    content: str,
    user_id: str
) -> None:
    """Extrahiere Wissen mit Transaction-Support"""
    try:
        functions = re.findall(r'def (\w+)\s*\(', content)
        classes = re.findall(r'class (\w+)', content)
        imports = re.findall(r'^(?:import|from)\s+(\w+)', content, re.MULTILINE)
        
        # Batch Insert für Performance
        knowledge_records = []
        
        for func in functions:
            knowledge_records.append(
                (user_id, f"user_{user_id}", 'defines_function', func, 0.9)
            )
        
        for cls in classes:
            knowledge_records.append(
                (user_id, f"user_{user_id}", 'defines_class', cls, 0.9)
            )
        
        for imp in imports:
            knowledge_records.append(
                (user_id, f"user_{user_id}", 'imports_library', imp, 0.7)
            )
        
        if knowledge_records:
            await conn.executemany("""
                INSERT INTO knowledge_graph 
                (user_id, subject, predicate, object, confidence)
                VALUES ($1, $2, $3, $4, $5)
                ON CONFLICT DO NOTHING
            """, knowledge_records)
            
    except Exception as e:
        logger.warning(f"Wissens-Extraktion fehlgeschlagen: {e}")
        raise  # Re-raise für Transaction Rollback

# API Endpoints
@app.on_event("startup")
async def startup_event():
    """Starte Event: Lade Modelle und DB"""
    try:
        load_embeddings_model()
        await init_database()
        logger.info("Cognee erfolgreich gestartet")
    except Exception as e:
        logger.error(f"Startup fehlgeschlagen: {e}")
        raise

# ✅ RICHTIG - Mit Transaction
@app.post("/memory/store", status_code=201)
async def store_memory(request: MemoryRequest):
    """Speichere Memory atomar mit Knowledge Graph"""
    try:
        # Generiere Embedding außerhalb Transaction (kann lange dauern)
        embedding = generate_embedding(request.content)
        
        async with db_pool.acquire() as conn:
            async with conn.transaction():  # ✅ Transaction
                # Speichere Memory
                row = await conn.fetchrow("""
                    INSERT INTO memories (user_id, content, embedding, metadata)
                    VALUES ($1, $2, $3, $4)
                    RETURNING id, created_at
                """, request.user_id, request.content, embedding, request.metadata)
                
                memory_id = row['id']
                created_at = row['created_at']
                
                # Extrahiere und speichere Knowledge im selben Transaction
                await extract_knowledge_transactional(
                    conn, request.content, request.user_id
                )
        
        logger.info(f"Memory gespeichert (ID: {memory_id})")
        
        return {
            "status": "success",
            "memory_id": memory_id,
            "created_at": created_at.isoformat()
        }
    except Exception as e:
        logger.error(f"Speichern fehlgeschlagen: {e}")
        raise HTTPException(status_code=500, detail="Failed to store memory")

@app.get("/memory/search")
async def search_memory(query: str, user_id: Optional[str] = None, limit: int = 10):
    """Semantic Suche in Memories"""
    try:
        logger.info(f"Suche nach: {query[:50]}...")
        
        # Generiere Query-Embedding
        query_embedding = generate_embedding(query)
        
        # Suche in DB
        async with db_pool.acquire() as conn:
            if user_id:
                rows = await conn.fetch("""
                    SELECT id, content, metadata, created_at,
                           1 - (embedding <=> $1::vector) as similarity
                    FROM memories
                    WHERE user_id = $2
                    ORDER BY embedding <=> $1::vector
                    LIMIT $3
                """, query_embedding, user_id, limit)
            else:
                rows = await conn.fetch("""
                    SELECT id, content, metadata, created_at,
                           1 - (embedding <=> $1::vector) as similarity
                    FROM memories
                    ORDER BY embedding <=> $1::vector
                    LIMIT $2
                """, query_embedding, limit)
        
        memories = []
        for row in rows:
            memories.append(MemoryResponse(
                id=row['id'],
                content=row['content'],
                metadata=row['metadata'],
                similarity=float(row['similarity']),
                created_at=row['created_at']
            ))
        
        logger.info(f"{len(memories)} Memories gefunden")
        
        return {"memories": [m.dict() for m in memories], "query": query}
        
    except Exception as e:
        logger.error(f"Suche fehlgeschlagen: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ✅ RICHTIG - Mit Pagination
@app.get("/memory/user/{user_id}")
async def get_user_memories(
    user_id: str,
    limit: int = 50,
    offset: int = 0,  # ✅ NEU
    order_by: str = "created_at"  # ✅ NEU
):
    """Hole Memories mit Pagination"""
    try:
        async with db_pool.acquire() as conn:
            # Count total
            total = await conn.fetchval(
                "SELECT COUNT(*) FROM memories WHERE user_id = $1",
                user_id
            )
            
            # Validate order_by to prevent SQL injection
            allowed_columns = {"created_at", "updated_at", "id"}
            if order_by not in allowed_columns:
                order_by = "created_at"
            
            # Fetch page
            rows = await conn.fetch(f"""
                SELECT id, content, metadata, created_at
                FROM memories
                WHERE user_id = $1
                ORDER BY {order_by} DESC
                LIMIT $2 OFFSET $3
            """, user_id, limit, offset)
        
        memories = []
        for row in rows:
            memories.append({
                "id": row['id'],
                "content": row['content'],
                "metadata": row['metadata'],
                "created_at": row['created_at'].isoformat()
            })
        
        return {
            "user_id": user_id,
            "memories": memories,
            "pagination": {  # ✅ NEU
                "total": total,
                "limit": limit,
                "offset": offset,
                "has_more": offset + limit < total
            }
        }
        
    except Exception as e:
        logger.error(f"Abrufen fehlgeschlagen: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/memory/{memory_id}")
async def delete_memory(memory_id: int):
    """Lösche Memory"""
    try:
        async with db_pool.acquire() as conn:
            result = await conn.execute(
                "DELETE FROM memories WHERE id = $1", memory_id
            )
            
            if result == "DELETE 0":
                raise HTTPException(status_code=404, detail="Memory nicht gefunden")
        
        logger.info(f"Memory {memory_id} gelöscht")
        return {"status": "deleted", "memory_id": memory_id}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Löschen fehlgeschlagen: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/knowledge-graph/{user_id}")
async def get_knowledge_graph(user_id: str):
    """Hole Knowledge Graph für User"""
    try:
        async with db_pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT subject, predicate, object, confidence, created_at
                FROM knowledge_graph
                WHERE user_id = $1
                ORDER BY confidence DESC, created_at DESC
            """, user_id)
        
        edges = []
        for row in rows:
            edges.append({
                "subject": row['subject'],
                "predicate": row['predicate'],
                "object": row['object'],
                "confidence": row['confidence'],
                "created_at": row['created_at'].isoformat()
            })
        
        return {"user_id": user_id, "edges": edges}
        
    except Exception as e:
        logger.error(f"Knowledge Graph fehlgeschlagen: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health Check mit einheitlichem Format"""
    try:
        # Prüfe Datenbankverbindung
        db_status = db_pool is not None
        if db_pool:
            async with db_pool.acquire() as conn:
                await conn.execute("SELECT 1")
        
        return {
            "status": "healthy",
            "service": "cognee-memory-api",
            "version": "2.0.0",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "dependencies": {
                "database": db_status,
                "embeddings_model": embeddings_model is not None,
                "pgvector": True
            },
            "metrics": {
                "embedding_dim": EMBEDDING_DIM,
                "model_name": os.getenv("EMBEDDINGS_MODEL", "all-MiniLM-L6-v2")
            }
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "service": "cognee-memory-api",
            "version": "2.0.0",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "error": str(e)
        }, 503

@app.get("/")
async def root():
    """Root Endpoint"""
    return {
        "service": "Cognee Memory API",
        "version": "2.0.0",
        "embedding_dim": EMBEDDING_DIM,
        "model": os.getenv("EMBEDDINGS_MODEL", "all-MiniLM-L6-v2")
    }
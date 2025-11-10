import os
import logging
import json
import re
import asyncio
from typing import List, Optional, Dict, Any
from datetime import datetime, timezone
from functools import wraps
from collections import defaultdict
from time import time

import asyncpg
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator
from sentence_transformers import SentenceTransformer
from asyncpg.exceptions import UniqueViolationError
from fastapi.middleware.cors import CORSMiddleware

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("cognee")

app = FastAPI(
    title="Cognee Memory API",
    description="Vector memory storage for AI-generated code",
    version="2.0.0"
)

# CORS Middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("ALLOWED_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic Models
class MemoryRequest(BaseModel):
    user_id: str = Field(..., min_length=1, max_length=255, pattern=r'^[a-zA-Z0-9_-]+$')
    content: str = Field(..., min_length=10, max_length=100000)
    metadata: Dict[str, Any] = Field(default_factory=dict, max_items=50)
    
    @validator('content')
    def validate_content(cls, v):
        if len(v.strip()) < 10:
            raise ValueError('Content too short after stripping')
        return v.strip()
    
    @validator('metadata')
    def validate_metadata(cls, v):
        # Prevent oversized metadata (max 10KB when serialized)
        if len(json.dumps(v)) > 10000:
            raise ValueError('Metadata too large (max 10KB)')
        return v

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

# Configuration constants
EMBEDDING_DIM = 384  # Dimension for all-MiniLM-L6-v2 model
KNOWLEDGE_CONFIDENCE_THRESHOLD = 0.7

# Rate limiting configuration
RATE_LIMIT_MAX_REQUESTS = 100
RATE_LIMIT_WINDOW = 60  # seconds

# Whitelisted ORDER BY clauses to prevent SQL injection
# Each key maps to a validated ORDER BY clause
ALLOWED_ORDERS = {
    'created_at': 'created_at DESC',
    'updated_at': 'updated_at DESC',
    'id': 'id DESC'
}

def load_embeddings_model():
    """
    Load the SentenceTransformer model for generating text embeddings.
    Model name is configured via EMBEDDINGS_MODEL environment variable.
    """
    global embeddings_model
    
    model_name = os.getenv("EMBEDDINGS_MODEL", "all-MiniLM-L6-v2")
    logger.info(f"Loading embeddings model: {model_name}")
    
    embeddings_model = SentenceTransformer(model_name)
    logger.info("Embeddings model loaded successfully")

def generate_embedding(text: str) -> List[float]:
    """
    Generate vector embedding for the given text.
    Returns a list of floats (dimension depends on model).
    """
    if embeddings_model is None:
        raise RuntimeError("Embeddings model not loaded")
    
    # Generate embedding as numpy array, then convert to list for database storage
    embedding = embeddings_model.encode(text, convert_to_tensor=False)
    return embedding.tolist()

class RateLimiter:
    """
    Thread-safe rate limiter using sliding window algorithm.
    Tracks requests per endpoint and user separately.
    """
    def __init__(self):
        # Store request timestamps per rate key (endpoint:user)
        self.requests = defaultdict(list)
        # Use separate locks per rate key for concurrency safety
        self._locks = defaultdict(asyncio.Lock)
    
    async def is_allowed(self, key: str, endpoint: str, max_requests: int, window: int) -> bool:
        """
        Check if a request should be allowed based on rate limits.
        
        Args:
            key: Unique identifier for the user/client
            endpoint: Function name of the endpoint
            max_requests: Maximum requests allowed in the window
            window: Time window in seconds
        
        Returns:
            True if request is allowed, False if rate limit exceeded
        """
        rate_key = f"{endpoint}:{key}"
        
        async with self._locks[rate_key]:
            now = time()
            user_times = self.requests[rate_key]
            # Remove timestamps outside the current window (sliding window cleanup)
            user_times[:] = [t for t in user_times if now - t < window]
            
            if len(user_times) >= max_requests:
                return False
            
            user_times.append(now)
            return True

# Global rate limiter instance
rate_limiter = RateLimiter()

def rate_limit(max_requests: int = RATE_LIMIT_MAX_REQUESTS, window: int = RATE_LIMIT_WINDOW):
    """
    Decorator to apply rate limiting to FastAPI endpoints.
    Assumes the first argument is a Pydantic request model with a user_id field.
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Extract request object (first argument for FastAPI endpoints)
            request = kwargs.get('request') or args[0]
            user_id = request.user_id
            
            if not await rate_limiter.is_allowed(user_id, func.__name__, max_requests, window):
                raise HTTPException(
                    status_code=429,
                    detail=f"Rate limit exceeded: {max_requests} requests per {window}s allowed"
                )
            
            return await func(*args, **kwargs)
        return wrapper
    return decorator

async def init_database():
    """
    Initialize database connection pool and create necessary tables.
    Connection pool is configured for production use with reasonable limits.
    """
    global db_pool
    
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        raise ValueError("DATABASE_URL environment variable not set")
    
    # Create connection pool with production-ready settings
    try:
        db_pool = await asyncpg.create_pool(
            db_url,
            min_size=5,  # Minimum idle connections
            max_size=20,  # Maximum connections (realistic for most setups)
            command_timeout=60,  # Timeout for individual commands
            max_queries=50000,  # Max queries per connection before recycling
            max_inactive_connection_lifetime=300,  # Recycle idle connections after 5 minutes
            server_settings={
                'application_name': 'cognee-memory-api',
                'statement_timeout': '60000'  # 60s statement timeout at server level
            }
        )
        logger.info("Database connection pool created successfully")
    except Exception as e:
        logger.error(f"Failed to create database pool: {e}")
        raise RuntimeError("Database connection failed") from e
    
    # Create tables and indexes
    async with db_pool.acquire() as conn:
        # Enable pgvector extension for vector operations
        await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        
        # Create knowledge_graph table with unique constraint first
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS knowledge_graph (
                id SERIAL PRIMARY KEY,
                user_id VARCHAR(255),
                subject TEXT NOT NULL,
                predicate TEXT NOT NULL,
                object TEXT NOT NULL,
                confidence FLOAT DEFAULT 0.8,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(user_id, subject, predicate, object)
            )
        """)
        
        # Main memories table for storing text content and embeddings
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id SERIAL PRIMARY KEY,
                user_id VARCHAR(255) NOT NULL,
                content TEXT NOT NULL,
                embedding vector($1),  -- Vector for semantic search
                metadata JSONB DEFAULT '{}',  -- Flexible metadata storage
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """, EMBEDDING_DIM)
        
        # Create indexes for performance
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_memories_user_id ON memories(user_id)")
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_memories_created ON memories(created_at DESC)")
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_knowledge_user ON knowledge_graph(user_id)")
        
        logger.info("Database tables and indexes created successfully")

async def extract_knowledge_transactional(
    conn: asyncpg.Connection,
    content: str,
    user_id: str
) -> None:
    """
    Extract knowledge from code content and store in knowledge graph.
    This function is designed to be run inside a transaction.
    
    Extracts:
    - Function definitions
    - Class definitions
    - Library imports
    
    Each extraction is stored as a (subject, predicate, object) triple with confidence score.
    Uses UPSERT to update confidence if the triple already exists.
    """
    try:
        # Use regex to extract code elements
        functions = re.findall(r'def (\w+)\s*\(', content)
        classes = re.findall(r'class (\w+)', content)
        imports = re.findall(r'^(?:import|from)\s+(\w+)', content, re.MULTILINE)
        
        # Prepare batch insert for performance
        knowledge_records = []
        
        for func in functions:
            knowledge_records.append((user_id, f"user_{user_id}", 'defines_function', func, 0.9))
        
        for cls in classes:
            knowledge_records.append((user_id, f"user_{user_id}", 'defines_class', cls, 0.9))
        
        for imp in imports:
            knowledge_records.append((user_id, f"user_{user_id}", 'imports_library', imp, 0.7))
        
        # Bulk upsert knowledge triples using UNNEST for asyncpg compatibility
        # ON CONFLICT DO UPDATE sets confidence to the higher value and updates timestamp
        if knowledge_records:
            await conn.execute("""
                INSERT INTO knowledge_graph 
                (user_id, subject, predicate, object, confidence)
                SELECT * FROM UNNEST($1::text[], $2::text[], $3::text[], $4::text[], $5::float[])
                ON CONFLICT (user_id, subject, predicate, object)
                DO UPDATE SET 
                    confidence = GREATEST(knowledge_graph.confidence, EXCLUDED.confidence),
                    created_at = CURRENT_TIMESTAMP
            """, 
                [r[0] for r in knowledge_records],  # user_id
                [r[1] for r in knowledge_records],  # subject
                [r[2] for r in knowledge_records],  # predicate
                [r[3] for r in knowledge_records],  # object
                [r[4] for r in knowledge_records]   # confidence
            )
            
    except Exception as e:
        logger.warning(f"Knowledge extraction failed: {e}")
        raise  # Re-raise to trigger transaction rollback

# API Endpoints
@app.on_event("startup")
async def startup_event():
    """Application startup: load models and initialize database"""
    try:
        load_embeddings_model()
        await init_database()
        logger.info("Cognee API started successfully")
    except Exception as e:
        logger.error(f"Startup failed: {e}")
        raise

@app.post("/memory/store", status_code=201)
@rate_limit(max_requests=100, window=60)
async def store_memory(request: MemoryRequest):
    """
    Store a memory atomically with knowledge graph extraction.
    Embedding is generated before the transaction to minimize DB lock time.
    """
    # Generate embedding outside transaction (read-only, potentially slow)
    try:
        embedding = generate_embedding(request.content)
    except Exception as e:
        logger.error(f"Embedding generation failed: {e}")
        raise HTTPException(status_code=400, detail=f"Embedding generation failed: {e}")
    
    # Store memory and knowledge in a single transaction
    try:
        async with db_pool.acquire() as conn:
            async with conn.transaction():
                # Insert memory record
                row = await conn.fetchrow("""
                    INSERT INTO memories (user_id, content, embedding, metadata)
                    VALUES ($1, $2, $3, $4)
                    RETURNING id, created_at
                """, request.user_id, request.content, embedding, request.metadata)
                
                memory_id = row['id']
                created_at = row['created_at']
                
                # Extract and store knowledge in same transaction
                await extract_knowledge_transactional(
                    conn, request.content, request.user_id
                )
        
        logger.info(f"Memory stored successfully (ID: {memory_id})")
        
        return {
            "status": "success",
            "memory_id": memory_id,
            "created_at": created_at.isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to store memory: {e}")
        raise HTTPException(status_code=500, detail="Failed to store memory")

@app.get("/memory/search")
async def search_memory(query: str, user_id: Optional[str] = None, limit: int = 10):
    """
    Perform semantic search across memories using vector similarity.
    Results are ordered by cosine similarity (converted to distance for pgvector).
    """
    try:
        logger.info(f"Searching for: {query[:50]}...")
        
        # Generate query embedding
        query_embedding = generate_embedding(query)
        
        # Search in database using pgvector cosine similarity
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
        
        # Convert to response objects
        memories = [
            MemoryResponse(
                id=row['id'],
                content=row['content'],
                metadata=row['metadata'],
                similarity=float(row['similarity']),
                created_at=row['created_at']
            )
            for row in rows
        ]
        
        logger.info(f"Found {len(memories)} memories")
        
        return {"memories": [m.dict() for m in memories], "query": query}
        
    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/memory/user/{user_id}")
async def get_user_memories(
    user_id: str,
    limit: int = 50,
    offset: int = 0,
    order_by: str = "created_at"
):
    """
    Retrieve paginated memories for a specific user.
    Safe ORDER BY implementation prevents SQL injection.
    """
    try:
        # Validate order_by parameter against whitelist
        order_clause = ALLOWED_ORDERS.get(order_by.strip().lower())
        if not order_clause:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid order_by column. Allowed: {', '.join(sorted(ALLOWED_ORDERS.keys()))}"
            )

        async with db_pool.acquire() as conn:
            # Get total count for pagination metadata
            total = await conn.fetchval(
                "SELECT COUNT(*) FROM memories WHERE user_id = $1",
                user_id
            )
            
            # Use separate prepared statements for each allowed ordering
            # This maintains prepared statement safety while supporting multiple sort options
            if order_clause == 'created_at DESC':
                rows = await conn.fetch("""
                    SELECT id, content, metadata, created_at
                    FROM memories
                    WHERE user_id = $1
                    ORDER BY created_at DESC
                    LIMIT $2 OFFSET $3
                """, user_id, limit, offset)
            elif order_clause == 'updated_at DESC':
                rows = await conn.fetch("""
                    SELECT id, content, metadata, created_at
                    FROM memories
                    WHERE user_id = $1
                    ORDER BY updated_at DESC
                    LIMIT $2 OFFSET $3
                """, user_id, limit, offset)
            elif order_clause == 'id DESC':
                rows = await conn.fetch("""
                    SELECT id, content, metadata, created_at
                    FROM memories
                    WHERE user_id = $1
                    ORDER BY id DESC
                    LIMIT $2 OFFSET $3
                """, user_id, limit, offset)
        
        # Build response
        memories = [
            {
                "id": row['id'],
                "content": row['content'],
                "metadata": row['metadata'],
                "created_at": row['created_at'].isoformat()
            }
            for row in rows
        ]
        
        return {
            "user_id": user_id,
            "memories": memories,
            "pagination": {
                "total": total,
                "limit": limit,
                "offset": offset,
                "has_more": offset + limit < total
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to retrieve memories: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/memory/{memory_id}")
async def delete_memory(memory_id: int):
    """Delete a specific memory by ID"""
    try:
        async with db_pool.acquire() as conn:
            result = await conn.execute(
                "DELETE FROM memories WHERE id = $1", memory_id
            )
            
            if result == "DELETE 0":
                raise HTTPException(status_code=404, detail="Memory not found")
        
        logger.info(f"Memory {memory_id} deleted successfully")
        return {"status": "deleted", "memory_id": memory_id}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete memory: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/knowledge-graph/{user_id}")
async def get_knowledge_graph(user_id: str):
    """Retrieve knowledge graph edges for a specific user"""
    try:
        async with db_pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT subject, predicate, object, confidence, created_at
                FROM knowledge_graph
                WHERE user_id = $1
                ORDER BY confidence DESC, created_at DESC
            """, user_id)
        
        edges = [
            {
                "subject": row['subject'],
                "predicate": row['predicate'],
                "object": row['object'],
                "confidence": row['confidence'],
                "created_at": row['created_at'].isoformat()
            }
            for row in rows
        ]
        
        return {"user_id": user_id, "edges": edges}
        
    except Exception as e:
        logger.error(f"Knowledge graph retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """
    Comprehensive health check endpoint.
    Checks database connectivity and model loading status.
    """
    try:
        # Check database connection
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
                "model_name": os.getenv("EMBEDDINGS_MODEL", "all-MiniLM-L6-v2"),
                "rate_limit_window": f"{RATE_LIMIT_WINDOW}s",
                "rate_limit_max": RATE_LIMIT_MAX_REQUESTS
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
    """API root endpoint with service information"""
    return {
        "service": "Cognee Memory API",
        "version": "2.0.0",
        "embedding_dim": EMBEDDING_DIM,
        "model": os.getenv("EMBEDDINGS_MODEL", "all-MiniLM-L6-v2"),
        "features": ["rate_limiting", "transactional_knowledge", "sql_injection_protection"]
    }
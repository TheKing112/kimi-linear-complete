import asyncio
import hashlib
import json
import logging
import os
import time

import redis.asyncio as redis

# Logger konfigurieren
logger = logging.getLogger(__name__)


class RedisClient:
    """Singleton Redis client for distributed locking and request deduplication."""
    
    _instance = None
    _lock = None
    _initialized = False
    
    def __new__(cls):
        """Ensure singleton instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    async def initialize(self) -> None:
        """
        Initialize Redis client connection.
        Call this once at startup before using the client.
        """
        if self.__class__._initialized:
            logger.debug("Redis client already initialized.")
            return
        
        # Lazy lock initialization to avoid event loop issues at import time
        if self.__class__._lock is None:
            self.__class__._lock = asyncio.Lock()
        
        async with self.__class__._lock:
            # Double-check pattern for thread safety
            if self.__class__._initialized:
                return
            
            try:
                # Initialize Redis client
                # Note: redis.from_url is synchronous and returns a client instance
                self.client = redis.from_url(
                    os.getenv("REDIS_URL", "redis://localhost:6379"),
                    encoding="utf-8",
                    decode_responses=True,
                    max_connections=50,
                    socket_keepalive=True,
                    socket_connect_timeout=5,
                    socket_timeout=5,
                    retry_on_timeout=True,
                    health_check_interval=30
                )
                self.max_retries = 3
                self.__class__._initialized = True
                logger.info("Redis client initialized successfully.")
            except Exception as e:
                logger.error(f"Failed to initialize Redis client: {e}")
                raise
    
    async def acquire_lock(
        self,
        lock_name: str,
        timeout: int = 300,
        retry_delay: float = 0.1
    ) -> bool:
        """Acquire distributed lock with retry."""
        lock_key = f"lock:{lock_name}"
        
        for attempt in range(self.max_retries):
            try:
                acquired = await self.client.set(
                    lock_key,
                    "1",
                    nx=True,
                    ex=timeout
                )
                
                if acquired:
                    return True
                
                # Exponential backoff
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(retry_delay * (2 ** attempt))
            
            except Exception as e:
                logger.error(f"Lock acquisition error: {e}")
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(retry_delay)
        
        return False
    
    async def acquire_lock_with_timeout(
        self,
        lock_name: str,
        timeout: int = 300,
        acquire_timeout: int = 30
    ) -> bool:
        """Acquire lock with acquisition timeout."""
        lock_key = f"lock:{lock_name}"
        start_time = time.time()
        
        while time.time() - start_time < acquire_timeout:
            acquired = await self.client.set(
                lock_key,
                "1",
                nx=True,
                ex=timeout
            )
            
            if acquired:
                return True
            
            await asyncio.sleep(0.1)
        
        return False
    
    async def release_lock(self, lock_name: str):
        """Release lock."""
        await self.client.delete(f"lock:{lock_name}")
    
    async def is_duplicate_request(self, user_id: str, repo_url: str, prompt: str) -> bool:
        """Check if same request was processed recently."""
        request_hash = hashlib.sha256(f"{user_id}:{repo_url}:{prompt}".encode()).hexdigest()
        exists = await self.client.exists(f"autonomous:request:{request_hash}")
        return exists
    
    async def health_check(self) -> bool:
        """Check Redis connection health."""
        try:
            await self.client.ping()
            return True
        except Exception as e:
            logger.error(f"Redis health check failed: {e}")
            return False
    
    async def ensure_connection(self):
        """Ensure Redis is connected."""
        if not await self.health_check():
            logger.info("Redis connection lost. Reconnecting...")
            # redis.from_url is synchronous
            self.client = redis.from_url(
                os.getenv("REDIS_URL", "redis://localhost:6379"),
                encoding="utf-8",
                decode_responses=True,
                max_connections=50,
                socket_keepalive=True,
                socket_connect_timeout=5,
                socket_timeout=5,
                retry_on_timeout=True,
                health_check_interval=30
            )
            logger.info("Redis reconnected")
    
    async def mark_request_processed(
        self,
        user_id: str,
        repo_url: str,
        prompt: str,
        ttl: int = 3600
    ):
        """Mark request as processed with validation."""
        if ttl < 60:
            raise ValueError("TTL must be at least 60 seconds")
        if ttl > 86400:
            raise ValueError("TTL cannot exceed 24 hours")
        
        request_hash = hashlib.sha256(
            f"{user_id}:{repo_url}:{prompt}".encode()
        ).hexdigest()
        
        await self.client.set(
            f"autonomous:request:{request_hash}",
            json.dumps({
                "user_id": user_id,
                "repo_url": repo_url,
                "timestamp": time.time()
            }),
            ex=ttl
        )
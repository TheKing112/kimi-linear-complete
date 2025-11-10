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
            
            client = None
            try:
                # Initialize Redis client
                client = redis.from_url(
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
                
                # Test connection before marking as initialized
                await client.ping()
                
                self.client = client
                self.max_retries = 3
                self.__class__._initialized = True
                logger.info("Redis client initialized successfully.")
                
            except Exception as e:
                # Cleanup on failure
                if client:
                    await client.close()
                logger.error(f"Failed to initialize Redis client: {e}")
                raise
    
    async def close(self) -> None:
        """Explicit cleanup method for Redis client."""
        if hasattr(self, 'client') and self.client:
            try:
                await self.client.close()
                await self.client.connection_pool.disconnect()
                self.__class__._initialized = False
                logger.info("Redis client closed successfully.")
            except Exception as e:
                logger.warning(f"Error closing Redis client: {e}")
    
    def _ensure_initialized(self) -> None:
        """Ensure client is initialized before use."""
        if not hasattr(self, 'client') or not self.__class__._initialized:
            raise RuntimeError("Redis client not initialized. Call initialize() first.")
    
    async def acquire_lock(
        self,
        lock_name: str,
        timeout: int = 300,
        retry_delay: float = 0.1
    ) -> bool:
        """Acquire distributed lock with retry."""
        self._ensure_initialized()
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
        self._ensure_initialized()
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
    
    async def release_lock(self, lock_name: str) -> None:
        """Release lock."""
        self._ensure_initialized()
        await self.client.delete(f"lock:{lock_name}")
    
    async def is_duplicate_request(self, user_id: str, repo_url: str, prompt: str) -> bool:
        """Check if same request was processed recently."""
        self._ensure_initialized()
        request_hash = hashlib.sha256(f"{user_id}:{repo_url}:{prompt}".encode()).hexdigest()
        exists = await self.client.exists(f"autonomous:request:{request_hash}")
        return exists
    
    async def health_check(self) -> bool:
        """Check Redis connection health."""
        self._ensure_initialized()
        try:
            await self.client.ping()
            return True
        except Exception as e:
            logger.error(f"Redis health check failed: {e}")
            return False
    
    async def ensure_connection(self) -> None:
        """Ensure Redis is connected, reconnect if necessary."""
        self._ensure_initialized()
        
        # Try health check first
        if await self.health_check():
            return
        
        logger.info("Redis connection lost. Reconnecting...")
        
        # Close old connection if exists
        if hasattr(self, 'client'):
            try:
                await self.client.close()
                await self.client.connection_pool.disconnect()
            except Exception as e:
                logger.warning(f"Error closing old Redis client: {e}")
        
        # Create new client
        client = None
        try:
            client = redis.from_url(
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
            
            # Test new connection
            await client.ping()
            
            self.client = client
            logger.info("Redis reconnected successfully.")
            
        except Exception as e:
            # Cleanup on failure
            if client:
                try:
                    await client.close()
                except Exception:
                    pass
            logger.error(f"Failed to reconnect Redis: {e}")
            raise
    
    async def mark_request_processed(
        self,
        user_id: str,
        repo_url: str,
        prompt: str,
        ttl: int = 3600
    ) -> None:
        """Mark request as processed with validation."""
        self._ensure_initialized()
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
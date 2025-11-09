# services/github-integration/src/redis_client.py
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
    def __init__(self):
        self.client = redis.from_url(
            os.getenv("REDIS_URL", "redis://localhost:6379")
        )
        self.max_retries = 3

    async def acquire_lock(
        self,
        lock_name: str,
        timeout: int = 300,
        retry_delay: float = 0.1
    ) -> bool:
        """Acquire distributed lock with retry"""
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
                
                # Kurze Verzögerung vor dem Retry
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(retry_delay * (2 ** attempt))
            
            except Exception as e:
                logger.error(f"Lock acquisition error: {e}")
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(retry_delay)
        
        return False

    async def release_lock(self, lock_name: str):
        """Release lock"""
        await self.client.delete(f"lock:{lock_name}")

    async def is_duplicate_request(self, user_id: str, repo_url: str, prompt: str) -> bool:
        """Check if same request was processed recently"""
        request_hash = hashlib.sha256(f"{user_id}:{repo_url}:{prompt}".encode()).hexdigest()
        exists = await self.client.exists(f"autonomous:request:{request_hash}")
        return exists

    async def health_check(self) -> bool:
        """Check Redis connection health"""
        try:
            await self.client.ping()
            return True
        except Exception as e:
            logger.error(f"Redis health check failed: {e}")
            return False

    async def ensure_connection(self):
        """Ensure Redis is connected"""
        if not await self.health_check():
            # Reconnect
            self.client = redis.from_url(
                os.getenv("REDIS_URL", "redis://localhost:6379")
            )
            logger.info("Redis reconnected")

    async def mark_request_processed(
        self,
        user_id: str,
        repo_url: str,
        prompt: str,
        ttl: int = 3600
    ):
        """Mark request as processed with validation"""
        # TTL validieren
        if ttl < 60 or ttl > 86400:
            raise ValueError("TTL must be between 60s and 24h")
        
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
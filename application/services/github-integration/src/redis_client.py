# services/github-integration/src/redis_client.py
import redis.asyncio as redis
import json
import hashlib
import os

class RedisClient:
    def __init__(self):
        self.client = redis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379"))
    
    async def acquire_lock(self, lock_name: str, timeout: int = 300) -> bool:
        """Acquire distributed lock for operation"""
        lock_key = f"lock:{lock_name}"
        acquired = await self.client.set(lock_key, "1", nx=True, ex=timeout)
        return acquired is True
    
    async def release_lock(self, lock_name: str):
        """Release lock"""
        await self.client.delete(f"lock:{lock_name}")
    
    async def is_duplicate_request(self, user_id: str, repo_url: str, prompt: str) -> bool:
        """Check if same request was processed recently"""
        request_hash = hashlib.sha256(f"{user_id}:{repo_url}:{prompt}".encode()).hexdigest()
        exists = await self.client.exists(f"autonomous:request:{request_hash}")
        return exists
    
    async def mark_request_processed(self, user_id: str, repo_url: str, prompt: str, ttl: int = 3600):
        """Mark request as processed with TTL"""
        request_hash = hashlib.sha256(f"{user_id}:{repo_url}:{prompt}".encode()).hexdigest()
        await self.client.set(f"autonomous:request:{request_hash}", "1", ex=ttl)
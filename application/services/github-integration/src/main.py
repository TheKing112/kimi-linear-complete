"""
GitHub Integration Service
- Repository Cloning & Analysis
- Integration mit Cognee Memory
- Webhook Handler für autonomes Coden
"""

import os
import logging
from typing import Optional, Dict, Any, List
from datetime import datetime, timezone
from collections import defaultdict
from time import time
import hmac
import hashlib

from fastapi import FastAPI, HTTPException, BackgroundTasks, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .github_client import GitHubClient
from .repo_analyzer import RepositoryAnalyzer
from .webhook_handler import WebhookHandler

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("github-integration")

app = FastAPI(
    title="GitHub Integration API",
    description="Autonomous GitHub integration for Kimi Linear",
    version="1.0.0"
)

# ✅ CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("ALLOWED_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Rate Limiter
class RateLimiter:
    def __init__(self):
        self.requests = defaultdict(list)
        self.limits = {
            "clone": (5, 3600),  # 5 per hour
            "analyze": (10, 3600),  # 10 per hour
            "webhook": (100, 60)  # 100 per minute
        }
    
    def is_allowed(self, key: str, endpoint: str) -> bool:
        now = time()
        limit, window = self.limits.get(endpoint, (10, 60))
        
        requests = self.requests[f"{endpoint}:{key}"]
        requests[:] = [req for req in requests if now - req < window]
        
        if len(requests) >= limit:
            return False
        
        requests.append(now)
        return True

rate_limiter = RateLimiter()

# Models
class RepoCloneRequest(BaseModel):
    repo_url: str = Field(..., description="GitHub Repository URL")
    branch: str = "main"
    user_id: str
    auto_analyze: bool = True

class AnalyzeRequest(BaseModel):
    repo_id: str
    user_id: str
    focus_paths: List[str] = Field(default_factory=list)

class WebhookPayload(BaseModel):
    ref: str
    repository: Dict[str, Any]
    commits: List[Dict[str, Any]]
    pusher: Dict[str, Any]

# Global instances
github_client: Optional[GitHubClient] = None
analyzer: Optional[RepositoryAnalyzer] = None
webhook_handler: Optional[WebhookHandler] = None

@app.on_event("startup")
async def startup_event():
    """Initialisiere GitHub Integration"""
    global github_client, analyzer, webhook_handler
    
    github_token = os.getenv("GITHUB_TOKEN")
    if not github_token:
        logger.warning("⚠️ GITHUB_TOKEN nicht gesetzt - Nur read-only Mode")
    
    github_client = GitHubClient(token=github_token)
    analyzer = RepositoryAnalyzer()
    webhook_handler = WebhookHandler()
    
    logger.info("🚀 GitHub Integration gestartet")

@app.post("/repo/clone", status_code=202)
async def clone_repository(
    request: RepoCloneRequest,
    background_tasks: BackgroundTasks
):
    """Klone Repository und starte Analyse im Hintergrund"""
    # ✅ Rate limit check
    if not rate_limiter.is_allowed(request.user_id, "clone"):
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded. Max 5 clones per hour."
        )
    
    try:
        repo_id = github_client.extract_repo_id(request.repo_url)
        
        # Hintergrund-Task für Cloning & Analyse
        background_tasks.add_task(
            analyzer.clone_and_analyze,
            request.repo_url,
            request.branch,
            request.user_id,
            request.auto_analyze
        )
        
        return {
            "status": "accepted",
            "repo_id": repo_id,
            "message": "Repository wird im Hintergrund geklont und analysiert"
        }
    except Exception as e:
        logger.error(f"Clone fehlgeschlagen: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ✅ Webhook signature verification
def verify_webhook_signature(payload: bytes, signature: str) -> bool:
    """Verify GitHub webhook signature"""
    secret = os.getenv("GITHUB_WEBHOOK_SECRET")
    if not secret:
        logger.warning("No webhook secret configured")
        return True  # Allow in dev
    
    expected = hmac.new(
        secret.encode(),
        payload,
        hashlib.sha256
    ).hexdigest()
    
    return hmac.compare_digest(f"sha256={expected}", signature)

@app.post("/webhook/github")
async def github_webhook(
    request: Request,  # ✅ Changed to Request
    x_github_event: Optional[str] = Header(None),
    x_hub_signature_256: Optional[str] = Header(None)
):
    """Empfange GitHub Webhooks (push, pull_request, etc.)"""
    # ✅ Verify signature
    body = await request.body()
    
    if x_hub_signature_256:
        if not verify_webhook_signature(body, x_hub_signature_256):
            raise HTTPException(status_code=401, detail="Invalid signature")
    
    payload = await request.json()
    
    try:
        event_type = x_github_event or "ping"
        
        if event_type == "push":
            # Autonome Verarbeitung neuer Commits
            await webhook_handler.handle_push(payload)
            return {"status": "processed"}
        elif event_type == "ping":
            return {"status": "pong"}
        else:
            logger.info(f"Nicht verarbeitetes Event: {event_type}")
            return {"status": "ignored"}
            
    except Exception as e:
        logger.error(f"Webhook Fehler: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health Check mit einheitlichem Format und tieferen Prüfungen"""
    try:
        global github_client, analyzer
        
        # Prüfe GitHub Auth & Rate Limit
        github_healthy = False
        rate_limit_info = None
        if github_client:
            rate_limit = await github_client.get_rate_limit()
            github_healthy = rate_limit is not None
            rate_limit_info = rate_limit
        
        # Prüfe Analyzer
        analyzer_healthy = analyzer is not None
        
        # Bestimme Gesamt-Status
        if github_healthy and analyzer_healthy:
            status = "healthy"
        elif github_client and not github_client.is_authenticated():
            status = "degraded"  # Funktioniert im read-only Modus
        else:
            status = "unhealthy"
        
        return {
            "status": status,
            "service": "github-integration-api",
            "version": "1.0.0",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "dependencies": {
                "github_api": github_healthy,
                "repository_analyzer": analyzer_healthy,
                "authenticated": github_client.is_authenticated() if github_client else False
            },
            "metrics": {
                "rate_limit": rate_limit_info,
                "cloned_repos": len(os.listdir("/tmp/repos")) if os.path.exists("/tmp/repos") else 0
            }
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "service": "github-integration-api",
            "version": "1.0.0",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "error": str(e)
        }, 503
    
@app.get("/")
async def root():
    """Root Endpoint"""
    return {
        "service": "GitHub Integration API",
        "version": "1.0.0",
        "features": ["repo-cloning", "auto-analysis", "webhooks", "cognee-sync"]
    }
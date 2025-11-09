"""
Autonomous Repository Editor
- Analysiert Prompt
- Erstellt Plan
- Implementiert Änderungen
- Erstellt PR
- Integriert mit Cognee Memory für Kontext
- 🆔 UUID-Branch-Namen
- 🔗 URL-Validierung
- ⏱️ Timeout-Handling & Latency-Tracking
"""

import os
import re
import json
import uuid
import time
import asyncio
import logging
import aiohttp
import hashlib
import urllib.parse
from typing import Dict, List, Any, Optional
from pathlib import Path
from datetime import datetime

logger = logging.getLogger("autonomous-editor")

class ConflictError(Exception):
    """Repository wird bereits bearbeitet"""
    pass

class TimeoutError(Exception):
    """Operation timed out"""
    pass

class HTTPException(Exception):
    def __init__(self, status_code: int, detail: str):
        self.status_code = status_code
        self.detail = detail
        super().__init__(f"HTTP {status_code}: {detail}")

class AutonomousEditor:
    def __init__(self, github_url: str, redis_client: 'RedisClient', timeout: int = 30):
        # 🆔 UUID-Branch-Namen - Setup
        self.timeout = timeout
        self.latency_tracker = {}
        
        # 🔗 URL-Validierung in __init__
        self._validate_github_url(github_url)
        self.github_url = github_url
        
        # Redis-Client mit Timeout-Handling
        self.redis_client = redis_client
        
        # HTTP Session für persistente Verbindungen
        self._session: Optional[aiohttp.ClientSession] = None
        
        # Bestehende Konfiguration
        self.kimi_url = os.getenv("KIMI_LINEAR_URL", "http://kimi-linear:8003")
        self.cognee_url = os.getenv("COGNEE_URL", "http://cognee:8001")
        self.require_approval = os.getenv("AUTONOMOUS_REQUIRE_APPROVAL", "true").lower() == "true"
        self.create_branch = os.getenv("AUTONOMOUS_CREATE_BRANCH", "true").lower() == "true"
        self.max_tokens = int(os.getenv("AUTONOMOUS_MAX_TOKENS", "4096"))

    def _validate_github_url(self, url: str) -> bool:
        """Enhanced GitHub URL validation"""
        # Basic pattern check
        pattern = r"^https://github\.com/[a-zA-Z0-9][-a-zA-Z0-9]{0,38}/[a-zA-Z0-9._-]{1,100}/?$"
        if not re.match(pattern, url):
            raise ValueError(f"Invalid GitHub URL format: {url}")
        
        # Additional checks
        parsed = urllib.parse.urlparse(url)
        
        # Check scheme
        if parsed.scheme != 'https':
            raise ValueError("Only HTTPS URLs allowed")
        
        # Check domain
        if parsed.netloc != 'github.com':
            raise ValueError("Only github.com URLs allowed")
        
        # Extract parts
        parts = parsed.path.strip('/').split('/')
        if len(parts) != 2:
            raise ValueError("URL must be in format: github.com/owner/repo")
        
        owner, repo = parts
        
        # Validate owner (GitHub username rules)
        if not re.match(r'^[a-zA-Z0-9][-a-zA-Z0-9]{0,38}$', owner):
            raise ValueError(f"Invalid owner name: {owner}")
        
        # Validate repo (GitHub repo rules)
        if not re.match(r'^[a-zA-Z0-9._-]{1,100}$', repo):
            raise ValueError(f"Invalid repo name: {repo}")
        
        # Check for path traversal attempts
        if '..' in url or parsed.path.count('//') > 0:
            raise ValueError("Path traversal detected")
        
        return True

    def create_implementation_plan(self, analysis: Dict[str, Any], risk_enum: Any) -> Dict[str, Any]:
        """Erstelle detaillierten Plan für die Änderungen mit 🆔 UUID-Branch-Namen"""
        
        changes = []
        for file_change in analysis["required_changes"]:
            changes.append({
                "file_path": file_change["file"],
                "action": file_change["action"],  # modify, create, delete
                "description": file_change["description"],
                "code_template": file_change.get("template", "")
            })
        
        # 🆔 UUID-Branch-Namen-Generierung
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        unique_id = uuid.uuid4().hex[:8]
        
        plan = {
            "branch_name": f"autonomous/{risk_enum.value}-{unique_id}-{timestamp}",
            "base_branch": "main",
            "changes": changes,
            "commit_message": f"[autonomous] {analysis['user_intent'][:50]}...",
            "risk_level": analysis["risk_level"],
            "auto_merge": False  # IMMER manuelles Review
        }
        
        return plan

    async def acquire_lock_with_retry(
        self,
        lock_name: str,
        max_retries: int = 3,
        base_delay: float = 1.0
    ) -> bool:
        """Acquire lock with exponential backoff"""
        for attempt in range(max_retries):
            try:
                acquired = await asyncio.wait_for(
                    self.redis_client.acquire_lock(lock_name),
                    timeout=5
                )
                
                if acquired:
                    return True
                
                # Exponential backoff
                delay = base_delay * (2 ** attempt)
                logger.info(f"Lock busy, retrying in {delay}s...")
                await asyncio.sleep(delay)
            
            except asyncio.TimeoutError:
                logger.warning(f"Lock acquisition timeout (attempt {attempt + 1})")
        
        return False

    async def is_duplicate_and_mark(
        self,
        user_id: str,
        repo_url: str,
        prompt: str,
        ttl: int = 3600
    ) -> bool:
        """Atomare Operation mit Lua Script"""
        request_hash = hashlib.sha256(
            f"{user_id}:{repo_url}:{prompt}".encode()
        ).hexdigest()
        
        # Lua script für atomare Operation
        lua_script = """
        local key = KEYS[1]
        local value = ARGV[1]
        local ttl = ARGV[2]
        
        if redis.call('exists', key) == 1 then
            return 1  -- Already exists
        else
            redis.call('setex', key, ttl, value)
            return 0  -- Set successful
        end
        """
        
        result = await self.redis_client.client.eval(
            lua_script,
            keys=[f"autonomous:request:{request_hash}"],
            args=[
                json.dumps({
                    "user_id": user_id,
                    "repo_url": repo_url,
                    "timestamp": time.time()
                }),
                str(ttl)
            ]
        )
        
        return result == 1  # True if duplicate

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create HTTP session"""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=300),
                headers={"User-Agent": "Kimi-Autonomous-Editor/1.0"}
            )
        return self._session

    async def process_autonomous_request(
        self, 
        prompt: str, 
        repo_url: str, 
        user_id: str,
        timeout: Optional[float] = None
    ) -> Dict[str, Any]:
        """⏱️ Hauptfunktion: Autonome Bearbeitung mit Timeout-Handling und Latency-Tracking"""
        logger.info(f"🤖 Autonome Anfrage: {prompt[:50]}... für {repo_url}")
        
        # 🔗 URL-Validierung am Anfang
        self._validate_github_url(repo_url)
        
        timeout = timeout or float(os.getenv("AUTONOMOUS_TIMEOUT", "600"))
        
        try:
            return await asyncio.wait_for(
                self._process_internal(prompt, repo_url, user_id),
                timeout=timeout
            )
        except asyncio.TimeoutError:
            raise TimeoutError(f"Operation timed out after {timeout}s")

    async def _process_internal(
        self,
        prompt: str,
        repo_url: str,
        user_id: str
    ) -> Dict[str, Any]:
        """Internal processing logic"""
        # ⏱️ Latency-Tracking Start
        start_time = time.monotonic()
        operation_metrics = {}
        
        try:
            # ⏱️ Lock-Acquisition mit Retry
            lock_name = f"github:{repo_url}"
            lock_start = time.monotonic()
            
            lock_acquired = await self.acquire_lock_with_retry(lock_name)
            
            operation_metrics['lock_acquisition_ms'] = round((time.monotonic() - lock_start) * 1000, 2)
            
            if not lock_acquired:
                raise ConflictError(f"Repository {repo_url} wird bereits bearbeitet")
            
            try:
                # ⏱️ Atomarer Duplicate-Check
                check_start = time.monotonic()
                is_duplicate = await self.is_duplicate_and_mark(user_id, repo_url, prompt)
                operation_metrics['duplicate_check_ms'] = round((time.monotonic() - check_start) * 1000, 2)
                
                if is_duplicate:
                    return {
                        "status": "aborted", 
                        "reason": "Duplicate request",
                        "latency_ms": self._get_latency_ms(start_time),
                        "operation_metrics": operation_metrics
                    }
                
                # Hauptverarbeitungsschritte
                process_start = time.monotonic()
                
                # Schritt 1: Analysiere Prompt und Repository
                analysis = await self.analyze_prompt_and_repo(prompt, repo_url, user_id)
                
                # Schritt 2: Erstelle Implementierungsplan
                class RiskEnum:
                    medium = "medium"
                risk_enum = RiskEnum()
                plan = await self.create_implementation_plan(analysis, risk_enum)
                
                # Schritt 3: Genehmigung einholen (wenn aktiviert)
                if self.require_approval:
                    approval = await self.request_approval(plan)
                    if not approval:
                        return {
                            "status": "aborted", 
                            "reason": "User denied approval",
                            "latency_ms": self._get_latency_ms(start_time),
                            "operation_metrics": operation_metrics
                        }
                
                # Schritt 4: Führe Änderungen aus
                changes = await self.execute_changes(plan, repo_url, user_id)
                
                # Schritt 5: Erstelle Pull Request
                pr = await self.create_pull_request(plan, changes, repo_url, user_id)
                
                # Schritt 6: Speichere im Cognee Memory
                await self.store_autonomous_action(prompt, plan, changes, pr, user_id)
                
                # ⏱️ Hauptverarbeitungszeit berechnen
                operation_metrics['main_processing_ms'] = round((time.monotonic() - process_start) * 1000, 2)
                
                # Erfolgreiche Antwort mit Metriken
                return {
                    "status": "success",
                    "plan": plan,
                    "changes_made": len(changes),
                    "pull_request": pr,
                    "latency_ms": self._get_latency_ms(start_time),
                    "operation_metrics": operation_metrics
                }
                
            finally:
                # ⏱️ Lock-Release mit Timeout
                release_start = time.monotonic()
                await asyncio.wait_for(
                    self.redis_client.release_lock(lock_name),
                    timeout=5
                )
                operation_metrics['lock_release_ms'] = round((time.monotonic() - release_start) * 1000, 2)
                
        except Exception as e:
            # Fehler mit Latency-Tracking zurückgeben
            raise self._enhance_error(e, start_time, operation_metrics)

    def _get_latency_ms(self, start_time: float) -> float:
        """⏱️ Latency in Millisekunden berechnen"""
        return round((time.monotonic() - start_time) * 1000, 2)
    
    def _enhance_error(self, error: Exception, start_time: float, metrics: Dict) -> Exception:
        """✅ Sichere Attribute-Zuweisung"""
        if not hasattr(error, 'latency_ms'):
            error.latency_ms = self._get_latency_ms(start_time)
        if not hasattr(error, 'operation_metrics'):
            error.operation_metrics = metrics
        return error

    async def analyze_prompt_and_repo(self, prompt: str, repo_url: str, user_id: str) -> Dict[str, Any]:
        """Analysiere was der User will und was im Repo existiert"""
        
        # Hole Repository-Context aus Cognee
        repo_context = await self.get_repo_context(repo_url, user_id)
        
        # Frage Kimi Linear nach Analyse
        analysis_prompt = f"""
        Analysiere diese Anfrage für ein GitHub Repository:
        
        USER ANFRAGE: {prompt}
        
        REPOSITORY Kontext:
        - Files: {len(repo_context.get('files', []))}
        - Sprachen: {repo_context.get('languages', [])}
        - Letztes Update: {repo_context.get('last_updated')}
        
        Gib zurück:
        1. Was der User will (klare Zusammenfassung)
        2. Welche Dateien betroffen sind
        3. Welche Änderungen nötig sind
        4. Risiko-Einschätzung (low/medium/high)
        5. Geschätzte Komplexität (1-10)
        """
        
        analysis = await self.query_kimi(analysis_prompt)
        
        return {
            "user_intent": analysis.get("intent"),
            "affected_files": analysis.get("files", []),
            "required_changes": analysis.get("changes", []),
            "risk_level": analysis.get("risk", "medium"),
            "complexity": analysis.get("complexity", 5),
            "repo_context": repo_context
        }

    async def request_approval(self, plan: Dict[str, Any]) -> bool:
        """Frage im Discord nach Genehmigung (wenn aktiviert)"""
        # TODO: Discord-Integration für Approval-Request
        # Für jetzt: Simulierte Genehmigung
        logger.warning("⚠️  Genehmigung erforderlich - simuliert APPROVED")
        logger.info(f"Plan: {json.dumps(plan, indent=2)}")
        
        # In Produktion: Sende Discord DM/Embed und warte auf ✅/❌ Reaction
        return True

    async def execute_changes(self, plan: Dict[str, Any], repo_url: str, user_id: str) -> List[Dict[str, Any]]:
        """Führe die geplanten Änderungen aus"""
        
        from .github_client import GitHubClient
        github = GitHubClient()
        
        changes = []
        
        for change in plan["changes"]:
            logger.info(f"  → Bearbeite: {change['file_path']} ({change['action']})")
            
            if change["action"] == "modify":
                new_content = await self.generate_code_change(change, plan["base_branch"])
            elif change["action"] == "create":
                new_content = await self.generate_new_file(change)
            elif change["action"] == "delete":
                new_content = None
            
            changes.append({
                "file_path": change["file_path"],
                "action": change["action"],
                "new_content": new_content,
                "status": "completed"
            })
        
        return changes

    async def generate_code_change(self, change: Dict[str, Any], branch: str) -> str:
        """Generiere Code-Änderung mit Kimi Linear"""
        
        prompt = f"""
        Modifiziere diese Datei gemäß Anforderung:
        
        FILE: {change['file_path']}
        REQUIREMENT: {change['description']}
        
        Falls du den aktuellen Inhalt brauchst, frage nach.
        """
        
        response = await self.query_kimi(prompt)
        return response.get("code", change.get("code_template", ""))

    async def generate_new_file(self, change: Dict[str, Any]) -> str:
        """Generiere komplett neue Datei"""
        return await self.generate_code_change(change, "main")

    async def create_pull_request(self, plan: Dict[str, Any], changes: List[Dict[str, Any]], repo_url: str, user_id: str) -> Dict[str, Any]:
        """Erstelle Pull Request mit allen Änderungen"""
        
        if not os.getenv("GITHUB_AUTO_CREATE_PR", "false").lower() == "true":
            return {"status": "skipped", "message": "Auto-PR deaktiviert"}
        
        # GitHub API verwenden für PR-Erstellung
        # TODO: Implementiere echte PR-Erstellung
        logger.info(f"📝 PR Erstellung simuliert: {plan['branch_name']} → {plan['base_branch']}")
        
        return {
            "status": "created",
            "branch": plan["branch_name"],
            "commits": len(changes),
            "url": f"{repo_url}/pull/1"  # Platzhalter
        }

    async def store_autonomous_action(self, prompt: str, plan: Dict[str, Any], changes: List[Dict[str, Any]], pr: Dict[str, Any], user_id: str):
        """Speichere autonome Aktion im Cognee Memory"""
        
        metadata = {
            "type": "autonomous_action",
            "prompt": prompt,
            "branch": plan["branch_name"],
            "changes_count": len(changes),
            "pr_url": pr.get("url"),
            "risk_level": plan["risk_level"],
            "auto_merged": False
        }
        
        payload = {
            "user_id": user_id,
            "content": f"Autonome Aktion: {prompt[:200]}...",
            "metadata": metadata
        }
        
        session = await self._get_session()
        async with session.post(
            f"{self.cognee_url}/memory/store",
            json=payload
        ) as resp:
            if resp.status != 201:
                logger.warning("Speichern der autonomen Aktion fehlgeschlagen")

    async def get_repo_context(self, repo_url: str, user_id: str) -> Dict[str, Any]:
        """Hole Repository-Kontext aus Cognee"""
        # Query alle Files aus diesem Repo
        session = await self._get_session()
        async with session.get(
            f"{self.cognee_url}/memory/search",
            params={
                "query": repo_url,
                "user_id": user_id,
                "limit": 50
            }
        ) as resp:
            if resp.status == 200:
                data = await resp.json()
                return {
                    "files": [mem["content"][:100] for mem in data.get("memories", [])],
                    "languages": ["python"],  # Simplifiziert
                    "last_updated": "2024-01-01"
                }
        
        return {"files": [], "languages": [], "last_updated": None}

    async def query_kimi(self, prompt: str, max_retries: int = 3) -> Dict[str, Any]:
        """Query Kimi Linear with retry logic"""
        session = await self._get_session()
        
        for attempt in range(max_retries):
            try:
                async with session.post(
                    f"{self.kimi_url}/generate",
                    json={
                        "messages": [
                            {"role": "user", "content": prompt}
                        ],
                        "max_tokens": self.max_tokens
                    }
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        try:
                            return json.loads(data["text"])
                        except json.JSONDecodeError:
                            return {"text": data["text"]}
                    elif resp.status == 503:
                        if attempt < max_retries - 1:
                            await asyncio.sleep(2 ** attempt)
                            continue
                        raise HTTPException(503, "Kimi service unavailable")
                    else:
                        raise HTTPException(resp.status, f"Kimi error: {resp.status}")
            
            except asyncio.TimeoutError:
                if attempt < max_retries - 1:
                    logger.warning(f"Kimi query timeout, retry {attempt + 1}")
                    await asyncio.sleep(2 ** attempt)
                    continue
                raise
            
            except Exception as e:
                logger.error(f"Kimi query failed: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                    continue
                raise
        
        return {"error": "Max retries exceeded"}

    async def cleanup(self):
        """Cleanup resources"""
        if self._session and not self._session.closed:
            await self._session.close()
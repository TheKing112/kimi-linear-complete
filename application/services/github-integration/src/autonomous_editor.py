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
import aiohttp  # ✅ NEU
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

class HTTPException(Exception):  # ✅ NEU
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
        
        # Bestehende Konfiguration
        self.kimi_url = os.getenv("KIMI_LINEAR_URL", "http://kimi-linear:8003")
        self.cognee_url = os.getenv("COGNEE_URL", "http://cognee:8001")
        self.require_approval = os.getenv("AUTONOMOUS_REQUIRE_APPROVAL", "true").lower() == "true"
        self.create_branch = os.getenv("AUTONOMOUS_CREATE_BRANCH", "true").lower() == "true"
        self.max_tokens = int(os.getenv("AUTONOMOUS_MAX_TOKENS", "4096"))

    def _validate_github_url(self, url: str) -> bool:
        """Validate GitHub URL with improved pattern"""
        # Basic pattern check
        pattern = r"^https://github\.com/[a-zA-Z0-9-_.]+/[a-zA-Z0-9-_.]+/?$"
        if not re.match(pattern, url):
            raise ValueError(f"Invalid GitHub URL: {url}")
        
        # ✅ NEU - Zusätzliche Checks
        if url.count('/') < 4:
            raise ValueError("URL must include owner and repo")
        
        parts = url.rstrip('/').split('/')
        owner, repo = parts[-2], parts[-1]
        
        if len(owner) < 1 or len(repo) < 1:
            raise ValueError("Owner and repo names cannot be empty")
        
        # Check for suspicious characters
        suspicious = ['..', '<', '>', '|', '&', ';']
        if any(char in url for char in suspicious):
            raise ValueError("URL contains suspicious characters")
        
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

    async def acquire_lock_with_retry(  # ✅ NEU
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

    async def process_autonomous_request(
        self, 
        prompt: str, 
        repo_url: str, 
        user_id: str
    ) -> Dict[str, Any]:
        """⏱️ Hauptfunktion: Autonome Bearbeitung mit Timeout-Handling und Latency-Tracking"""
        logger.info(f"🤖 Autonome Anfrage: {prompt[:50]}... für {repo_url}")
        
        # 🔗 URL-Validierung am Anfang
        self._validate_github_url(repo_url)
        
        # ⏱️ Latency-Tracking Start
        start_time = time.monotonic()
        operation_metrics = {}
        
        try:
            # ⏱️ Lock-Acquisition mit Retry
            lock_name = f"github:{repo_url}"
            lock_start = time.monotonic()
            
            lock_acquired = await self.acquire_lock_with_retry(lock_name)  # ✅ VERBESSERT
            
            operation_metrics['lock_acquisition_ms'] = round((time.monotonic() - lock_start) * 1000, 2)
            
            if not lock_acquired:
                raise ConflictError(f"Repository {repo_url} wird bereits bearbeitet")
            
            try:
                # ⏱️ Duplicate-Check mit Timeout
                check_start = time.monotonic()
                is_duplicate = await asyncio.wait_for(
                    self.redis_client.is_duplicate_request(user_id, repo_url, prompt),
                    timeout=5
                )
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
                # Annahme: risk_enum ist verfügbar - in echtem Code anpassen
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
                
                # ⏱️ Markierung als verarbeitet mit Timeout
                mark_start = time.monotonic()
                await asyncio.wait_for(
                    self.redis_client.mark_request_processed(user_id, repo_url, prompt),
                    timeout=5
                )
                operation_metrics['mark_processed_ms'] = round((time.monotonic() - mark_start) * 1000, 2)
                
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
                
        except asyncio.TimeoutError as e:
            raise TimeoutError(f"Operation timed out after {self.timeout}s: {str(e)}")
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
        
        # GitHub Client für Änderungen
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
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.cognee_url}/memory/store",
                json=payload
            ) as resp:
                if resp.status != 201:
                    logger.warning("Speichern der autonomen Aktion fehlgeschlagen")

    async def get_repo_context(self, repo_url: str, user_id: str) -> Dict[str, Any]:
        """Hole Repository-Kontext aus Cognee"""
        # Query alle Files aus diesem Repo
        async with aiohttp.ClientSession() as session:
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

    async def query_kimi(self, prompt: str, max_retries: int = 3) -> Dict[str, Any]:  # ✅ VERBESSERT
        """Query Kimi Linear with retry logic"""
        for attempt in range(max_retries):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        f"{self.kimi_url}/generate",
                        json={
                            "messages": [
                                {"role": "user", "content": prompt}
                            ],
                            "max_tokens": self.max_tokens
                        },
                        timeout=aiohttp.ClientTimeout(total=300)
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
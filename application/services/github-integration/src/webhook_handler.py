import os
import logging
from typing import Dict, Any
import aiohttp

logger = logging.getLogger("webhook-handler")

class WebhookHandler:
    """Verarbeite GitHub Webhooks für autonomes Coden"""
    
    async def handle_push(self, payload: Dict[str, Any]):
        """Verarbeite Push Events - Autonome Code-Analyse"""
        repo_full_name = payload["repository"]["full_name"]
        commits = payload["commits"]
        
        logger.info(f"🔔 Push Event: {repo_full_name} ({len(commits)} commits)")
        
        for commit in commits:
            await self.process_commit(commit, repo_full_name)
    
    async def process_commit(self, commit: Dict[str, Any], repo_name: str):
        """Verarbeite einzelnen Commit"""
        commit_id = commit["id"][:7]
        author = commit["author"]["name"]
        message = commit["message"]
        
        logger.info(f"  → Commit {commit_id} von {author}: {message[:50]}...")
        
        # Hier kannst du autonome Logik einfügen:
        # 1. Diff analysieren
        # 2. Code-Qualität prüfen
        # 3. Vorschläge generieren
        # 4. Pull Request erstellen (optional)
        
        # Beispiel: Autonome Code-Review
        if "[auto-review]" in message.lower():
            await self.trigger_code_review(commit, repo_name)
    
    async def trigger_code_review(self, commit: Dict[str, Any], repo_name: str):
        """Löse automatische Code-Review mit Kimi Linear aus"""
        kim_url = os.getenv("KIMI_LINEAR_URL", "http://kimi-linear:8003")
        
        # Hole Diff (simplifiziert)
        diff_url = commit["url"] + ".diff"
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{kim_url}/generate",
                    json={
                        "messages": [
                            {"role": "system", "content": "Du bist ein Senior Code Reviewer. Analysiere den Diff und gib konkrete Verbesserungsvorschläge."},
                            {"role": "user", "content": f"Diff: {diff_url}"}
                        ],
                        "max_tokens": 2048
                    }
                ) as resp:
                    if resp.status == 200:
                        review = await resp.json()
                        logger.info(f"Automatische Review generiert: {review['text'][:100]}...")
                        
                        # Hier könntest du einen PR-Kommentar erstellen
                        # await self.post_pr_comment(repo_name, review["text"])
        
        except Exception as e:
            logger.error(f"Automatische Review fehlgeschlagen: {e}")
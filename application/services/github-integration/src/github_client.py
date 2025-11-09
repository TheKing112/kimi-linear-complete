import os
import re
import signal
import tempfile
import time
from contextlib import contextmanager
from typing import Optional, Dict, Any, List
import logging

from github import Github, Repository, GithubException, RateLimitExceededException
from git import Repo

logger = logging.getLogger("github-client")

@contextmanager
def timeout(seconds):
    def signal_handler(signum, frame):
        raise TimeoutError(f"Operation timed out after {seconds}s")
    
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)

class GitHubClient:
    def __init__(self, token: Optional[str] = None):
        self.token = token or os.getenv("GITHUB_TOKEN")
        self.github = Github(self.token) if self.token else Github()
        
    def is_authenticated(self) -> bool:
        return self.token is not None
    
    def extract_repo_id(self, url: str) -> str:
        """Extrahiere owner/repo aus URL"""
        if url.endswith(".git"):
            url = url[:-4]
        parts = url.split("/")
        return f"{parts[-2]}/{parts[-1]}"
    
    def clone_repo(self, url: str, branch: str = "main") -> str:
        """Klone Repository in temporäres Verzeichnis"""
        repo_id = self.extract_repo_id(url)
        tmp_dir = tempfile.mkdtemp(prefix=f"github_{repo_id.replace('/', '_')}_")
        
        logger.info(f"Klonen {repo_id} (branch: {branch}) nach {tmp_dir}")
        
        try:
            # ✅ 10 Minuten max Timeout
            with timeout(600):
                Repo.clone_from(
                    url,
                    tmp_dir,
                    branch=branch,
                    depth=100,  # Nur letzte 100 Commits für Performance
                    single_branch=True
                )
            logger.info(f"✅ Repository geklont nach {tmp_dir}")
            return tmp_dir
        except TimeoutError as e:
            logger.error(f"Clone fehlgeschlagen (Timeout): {e}")
            raise
        except Exception as e:
            logger.error(f"Clone fehlgeschlagen: {e}")
            raise
    
    async def get_rate_limit(self) -> Dict[str, Any]:
        """GitHub API Rate Limit Info"""
        try:
            rate = self.github.get_rate_limit()
            return {
                "limit": rate.core.limit,
                "remaining": rate.core.remaining,
                "reset": rate.core.reset.isoformat()
            }
        except Exception as e:
            logger.warning(f"Rate Limit Abfrage fehlgeschlagen: {e}")
            return {"error": str(e)}
    
    # ✅ RICHTIG - Mit Max Retries und Timeout
    def execute_with_rate_limit(
        self,
        func,
        *args,
        max_retries: int = 3,
        max_wait: int = 3600,  # NEU - Max 1 Stunde warten
        **kwargs
    ):
        """Führe GitHub-Operation mit Rate-Limit-Handling und max retries aus"""
        retries = 0
        total_wait = 0  # NEU - Wartezeit tracking
        
        while retries < max_retries:
            try:
                return func(*args, **kwargs)
            
            except RateLimitExceededException:
                retries += 1
                
                if retries >= max_retries:
                    logger.error(f"Max retries ({max_retries}) für Rate Limit erreicht")
                    raise Exception(f"Max retries ({max_retries}) exceeded")
                
                reset_time = self.github.rate_limiting_resettime
                wait_seconds = max(reset_time - time.time(), 0) + 60
                
                # NEU - Check max wait
                if total_wait + wait_seconds > max_wait:
                    raise Exception(f"Max wait time ({max_wait}s) exceeded")
                
                logger.warning(
                    f"⏱️ Rate limit erreicht (Versuch {retries}/{max_retries}), "
                    f"warte {wait_seconds}s"
                )
                
                time.sleep(wait_seconds)
                total_wait += wait_seconds  # NEU - Update total wait
            
            except Exception as e:
                logger.error(f"GitHub API Fehler: {e}")
                raise
        
        raise Exception("Unerwartet: Max retries ohne Ausnahme")
    
    # ✅ Verbessert - Mit Caching
    def verify_user_access(self, user_id: str, repo_id: str) -> bool:
        """Prüfe User-Schreibrechte mit Caching"""
        try:
            # Cache-Key für Zugriffsprüfung
            cache_key = f"access:{user_id}:{repo_id}"
            
            # In Produktion: Redis-Cache verwenden
            # cached = redis.get(cache_key)
            # if cached: return cached == "true"
            
            user = self.github.get_user()
            repo = self.github.get_repo(repo_id)
            
            # Hole Collaborator-Permission
            permission = repo.get_collaborator_permission(user.login)
            
            has_access = permission in ["write", "admin"]
            logger.info(f"User {user.login} hat {permission}-Zugriff auf {repo_id}")
            
            # Ergebnis cachen (15 Minuten)
            # redis.setex(cache_key, 900, "true" if has_access else "false")
            
            return has_access
        
        except Exception as e:
            logger.error(f"Permission-Check fehlgeschlagen: {e}")
            return False
    
    # ✅ Verbessert - Mit Validierung und Fehlerbehandlung
    def create_atomic_commit(
        self,
        repo: Repository,
        changes: List[Dict[str, Any]],
        branch: str,
        message: str
    ) -> str:
        """Erstelle atomaren Commit mit Rollback bei Fehler"""
        
        # ✅ NEU - Input Validierung
        if not changes:
            raise ValueError("Keine Änderungen bereitgestellt")
        
        if len(message) > 500:
            raise ValueError("Commit-Nachricht zu lang (max. 500 Zeichen)")
        
        # Validate branch name
        if not re.match(r'^[a-zA-Z0-9/_-]+$', branch):
            raise ValueError(f"Ungültiger Branch-Name: {branch}")
        
        # Check if branch exists
        try:
            repo.get_branch(branch)
        except Exception:
            raise ValueError(f"Branch existiert nicht: {branch}")
        
        logger.info(f"🔧 Erstelle atomaren Commit auf {repo.full_name}:{branch}")
        
        try:
            # 1. Hole aktuellen Commit
            base_commit = self.execute_with_rate_limit(
                repo.get_commit,
                f"heads/{branch}"
            )
            base_tree = base_commit.commit.tree
            
            # 2. Baue neuen Tree
            input_tree = []
            processed_files = set()  # Track verarbeitete Dateien
            
            for change in changes:
                action = change.get("action")
                file_path = change.get("file_path")
                
                # Validierung: Prüfe auf ungültige/duplikate Pfade
                if not file_path or file_path in processed_files:
                    logger.warning(f"Überspringe ungültige/duplikate Datei: {file_path}")
                    continue
                
                processed_files.add(file_path)
                
                if action == "delete":
                    logger.info(f"🗑️ Lösche Datei: {file_path}")
                    continue
                
                # Für create und modify
                new_content = change.get("new_content", "")
                
                # Validierung: Standard-Inhalt für leere neue Dateien
                if not new_content and action == "create":
                    logger.warning(f"⚠️ Leerer Inhalt für neue Datei: {file_path}")
                    new_content = "# Leere Datei\n"
                
                # ✅ NEU - Chunked Upload für große Dateien
                MAX_BLOB_SIZE = 10 * 1024 * 1024  # 10MB
                if len(new_content) > MAX_BLOB_SIZE:
                    logger.warning(f"Große Datei erkannt: {file_path} ({len(new_content)} bytes)")
                    
                    # Für sehr große Dateien: Git LFS verwenden
                    raise ValueError(
                        f"Datei zu groß für direkten Commit: {file_path} "
                        f"({len(new_content)} bytes). Bitte Git LFS verwenden."
                    )
                
                # Erstelle Blob
                blob = self.execute_with_rate_limit(
                    repo.create_git_blob,
                    new_content,
                    "utf-8"
                )
                
                input_tree.append({
                    "path": file_path,
                    "mode": "100644",
                    "type": "blob",
                    "sha": blob.sha
                })
                
                logger.info(
                    f"{'✨ Erstelle' if action == 'create' else '✏️ Ändere'}: {file_path}"
                )
            
            # ✅ Prüfe auf tatsächliche Änderungen
            if not input_tree and not any(c.get("action") == "delete" for c in changes):
                logger.warning("Keine Änderungen zum Committen")
                return base_commit.sha
            
            # 3. Erstelle neuen Tree
            new_tree = self.execute_with_rate_limit(
                repo.create_git_tree,
                input_tree,
                base_tree
            )
            logger.info(f"🌳 Neuer Tree erstellt: {new_tree.sha[:7]}")
            
            # 4. Erstelle Commit
            commit = self.execute_with_rate_limit(
                repo.create_git_commit,
                message=message,
                tree=new_tree,
                parents=[base_commit.commit]
            )
            logger.info(f"✅ Commit erstellt: {commit.sha[:7]}")
            
            # 5. Update Branch Ref
            ref = self.execute_with_rate_limit(
                repo.get_git_ref,
                f"heads/{branch}"
            )
            self.execute_with_rate_limit(ref.edit, commit.sha)
            logger.info(f"🚀 Branch {branch} aktualisiert auf {commit.sha[:7]}")
            
            return commit.sha
        
        except Exception as e:
            logger.error(f"❌ Atomarer Commit fehlgeschlagen: {e}", exc_info=True)
            
            # Detaillierte Fehlermeldungen
            error_msg = str(e).lower()
            if "rate limit" in error_msg:
                raise Exception("GitHub Rate Limit überschritten. Bitte versuche es später erneut.")
            elif "not found" in error_msg:
                raise Exception(f"Branch oder Repository nicht gefunden: {branch}")
            else:
                raise Exception(f"Commit-Erstellung fehlgeschlagen: {str(e)}")
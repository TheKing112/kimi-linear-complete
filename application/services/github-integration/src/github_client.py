import os
import time
from github import Github, Repository, GithubException, RateLimitExceededException
from git import Repo
import tempfile
from typing import Optional, Dict, Any, List
import logging

logger = logging.getLogger("github-client")

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
            Repo.clone_from(
                url,
                tmp_dir,
                branch=branch,
                depth=100,  # Nur letzte 100 Commits für Performance
                single_branch=True
            )
            logger.info(f"✅ Repository geklont nach {tmp_dir}")
            return tmp_dir
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
    
    # ⚡ Rate-Limit-Handler mit automatischem Retry
    def execute_with_rate_limit(self, func, *args, **kwargs):
        """Führe GitHub-Operation mit automatischem Retry bei Rate Limit aus"""
        while True:
            try:
                return func(*args, **kwargs)
            except RateLimitExceededException:
                reset_time = self.github.rate_limiting_resettime
                wait_seconds = max(reset_time - time.time(), 0) + 60
                logger.warning(f"⏱️ Rate limit hit, waiting {wait_seconds}s")
                time.sleep(wait_seconds)
            except Exception as e:
                logger.error(f"GitHub API Fehler: {e}")
                raise
    
    # 🔒 User-Permission-Check für Schreibrechte
    def verify_user_access(self, user_id: str, repo_id: str) -> bool:
        """Prüfe ob User Schreibrechte auf Repository hat"""
        try:
            # Prüfe ob Token zu User gehört
            user = self.github.get_user()
            repo = self.github.get_repo(repo_id)
            
            # Hole Collaborator-Permission
            permission = repo.get_collaborator_permission(user.login)
            
            # Nur WRITE oder ADMIN erlaubt
            return permission in ["write", "admin"]
        except Exception as e:
            logger.error(f"Permission-Check fehlgeschlagen: {e}")
            return False
    
    # 🔧 NEU: Atomarer Commit mit allen Änderungen
    def create_atomic_commit(self, repo: Repository, changes: List[Dict[str, Any]], branch: str, message: str) -> str:
        """
        Erstelle einen einzelnen atomaren Commit mit allen Änderungen
        
        Args:
            repo: GitHub Repository Objekt
            changes: Liste von Change-Dictionaries mit:
                - action: "create", "modify" oder "delete"
                - file_path: Pfad zur Datei
                - new_content: Neuer Inhalt (für create/modify)
                - sha: Aktuelle Blob SHA (optional, für modify)
            branch: Zielbranch name
            message: Commit Message
        
        Returns:
            SHA des erstellten Commits
        """
        logger.info(f"🔧 Erstelle atomaren Commit auf {repo.full_name}:{branch} mit {len(changes)} Änderungen")
        
        try:
            # 1. Hole aktuellen Commit und Tree
            base_commit = self.execute_with_rate_limit(repo.get_commit, f"heads/{branch}")
            base_tree = base_commit.commit.tree
            
            # 2. Baue neuen Tree
            input_tree = []
            for change in changes:
                action = change.get("action")
                file_path = change.get("file_path")
                
                if action == "delete":
                    # Datei wird ausgelassen -> gelöscht
                    logger.info(f"🗑️ Lösche Datei: {file_path}")
                    continue
                
                # Für create und modify
                new_content = change.get("new_content", "")
                if not new_content and action == "create":
                    logger.warning(f"⚠️  Leerer Inhalt für neue Datei: {file_path}")
                
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
                
                logger.info(f"{'✨ Erstelle' if action == 'create' else '✏️  Ändere'} Datei: {file_path}")
            
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
            ref = self.execute_with_rate_limit(repo.get_git_ref, f"heads/{branch}")
            self.execute_with_rate_limit(ref.edit, commit.sha)
            logger.info(f"🚀 Branch {branch} auf {commit.sha[:7]} aktualisiert")
            
            return commit.sha
            
        except Exception as e:
            logger.error(f"❌ Atomarer Commit fehlgeschlagen: {e}")
            raise
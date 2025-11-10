import os
import re
import time
import tempfile
import logging
from typing import Optional, Dict, Any, List
import concurrent.futures
import shutil

from github import Github, Repository, GithubException, RateLimitExceededException
from git import Repo

logger = logging.getLogger(__name__)


class GitHubConfig:
    """Konfiguration für GitHub-Operationen"""
    CLONE_TIMEOUT = 600  # 10 Minuten
    CLONE_DEPTH = 100
    MAX_BLOB_SIZE = 10 * 1024 * 1024  # 10MB
    MAX_COMMIT_MSG_LENGTH = 500
    ALLOWED_BRANCH_PATTERN = re.compile(r'^[a-zA-Z0-9/_-]+$')
    ALLOWED_ACTIONS = {"create", "modify", "delete"}
    ALLOWED_PERMISSIONS = {"write", "admin"}


class GitHubClient:
    def __init__(self, token: Optional[str] = None):
        """Initialisiert GitHub-Client mit Token (optional)"""
        self.token = token or os.getenv("GITHUB_TOKEN")
        self.github = Github(self.token) if self.token else Github()
        self.config = GitHubConfig()
    
    def is_authenticated(self) -> bool:
        """Prüft, ob ein GitHub Token verfügbar ist"""
        return self.token is not None
    
    def extract_repo_id(self, url: str) -> str:
        """Extrahiert owner/repo aus einer GitHub-URL"""
        if url.endswith(".git"):
            url = url[:-4]
        parts = url.split("/")
        return f"{parts[-2]}/{parts[-1]}"
    
    def clone_repo(self, url: str, branch: str = "main") -> str:
        """Klont Repository in temporäres Verzeichnis mit plattformübergreifendem Timeout"""
        repo_id = self.extract_repo_id(url)
        tmp_dir = tempfile.mkdtemp(prefix=f"github_{repo_id.replace('/', '_')}_")
        
        logger.info(f"Klonen {repo_id} (branch: {branch}) nach {tmp_dir}")
        
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(
                    Repo.clone_from,
                    url,
                    tmp_dir,
                    branch=branch,
                    depth=self.config.CLONE_DEPTH,
                    single_branch=True
                )
                try:
                    future.result(timeout=self.config.CLONE_TIMEOUT)
                except concurrent.futures.TimeoutError:
                    raise TimeoutError(f"Clone-Operation timed out after {self.config.CLONE_TIMEOUT}s")
            
            logger.info(f"✅ Repository geklont nach {tmp_dir}")
            return tmp_dir
            
        except Exception as e:
            logger.error(f"Clone fehlgeschlagen: {e}")
            self._cleanup_tmp_dir(tmp_dir)
            raise
    
    def _cleanup_tmp_dir(self, tmp_dir: str):
        """Helfer-Methode zum Aufräumen temporärer Verzeichnisse"""
        try:
            shutil.rmtree(tmp_dir, ignore_errors=True)
        except Exception as e:
            logger.warning(f"Temp-Verzeichnis Cleanup fehlgeschlagen: {e}")

    def get_rate_limit(self) -> Dict[str, Any]:
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
    
    def execute_with_rate_limit(
        self,
        func,
        *args,
        max_retries: int = 3,
        max_wait: int = 3600,
        **kwargs
    ):
        """Führt GitHub-Operation mit Rate-Limit-Handling und Max-Retries aus"""
        retries = 0
        total_wait = 0
        
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
                
                if total_wait + wait_seconds > max_wait:
                    raise Exception(f"Max wait time ({max_wait}s) exceeded")
                
                logger.warning(
                    f"⏱️ Rate limit erreicht (Versuch {retries}/{max_retries}), "
                    f"warte {wait_seconds}s"
                )
                
                time.sleep(wait_seconds)
                total_wait += wait_seconds
            
            except GithubException as e:
                logger.error(f"GitHub API Fehler: {e}")
                raise
            except Exception as e:
                logger.error(f"Unerwarteter Fehler: {e}")
                raise
        
        raise Exception("Unerwartet: Max retries ohne Ausnahme")
    
    def verify_user_access(self, user_id: str, repo_id: str) -> bool:
        """Prüft User-Schreibrechte (mit Caching-Platzhalter)"""
        CACHE_KEY_PREFIX = "access"
        cache_key = f"{CACHE_KEY_PREFIX}:{user_id}:{repo_id}"
        
        try:
            user = self.github.get_user()
            repo = self.github.get_repo(repo_id)
            
            permission = repo.get_collaborator_permission(user.login)
            
            has_access = permission in self.config.ALLOWED_PERMISSIONS
            logger.info(f"User {user.login} hat {permission}-Zugriff auf {repo_id}")
            
            return has_access
        
        except GithubException as e:
            logger.error(f"Permission-Check fehlgeschlagen: {e}")
            return False
        except Exception as e:
            logger.error(f"Unerwarteter Fehler bei Permission-Check: {e}")
            return False
    
    def create_atomic_commit(
        self,
        repo: Repository,
        changes: List[Dict[str, Any]],
        branch: str,
        message: str
    ) -> str:
        """Erstellt atomaren Commit mit Rollback bei Fehlern"""
        
        if not changes:
            raise ValueError("Keine Änderungen bereitgestellt")
        
        if len(message) > self.config.MAX_COMMIT_MSG_LENGTH:
            raise ValueError(f"Commit-Nachricht zu lang (max. {self.config.MAX_COMMIT_MSG_LENGTH} Zeichen)")
        
        if not self.config.ALLOWED_BRANCH_PATTERN.match(branch):
            raise ValueError(f"Ungültiger Branch-Name: {branch}")
        
        try:
            repo.get_branch(branch)
        except GithubException:
            raise ValueError(f"Branch existiert nicht: {branch}")
        except Exception as e:
            logger.error(f"Fehler beim Überprüfen des Branches: {e}")
            raise
        
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
            processed_files = set()
            
            for change in changes:
                action = change.get("action")
                file_path = change.get("file_path")
                
                if action not in self.config.ALLOWED_ACTIONS:
                    raise ValueError(f"Ungültige Aktion '{action}'. Erlaubt: {self.config.ALLOWED_ACTIONS}")
                
                if not file_path or not file_path.strip():
                    logger.warning(f"Überspringe ungültigen Datei-Pfad: {file_path}")
                    continue
                
                file_path = file_path.strip("/")
                
                if file_path in processed_files:
                    logger.warning(f"Überspringe duplikate Datei: {file_path}")
                    continue
                
                processed_files.add(file_path)
                
                if action == "delete":
                    logger.info(f"🗑️ Lösche Datei: {file_path}")
                    continue
                
                new_content = change.get("new_content", "")
                
                if not new_content and action == "create":
                    logger.warning(f"⚠️ Leerer Inhalt für neue Datei: {file_path}")
                    new_content = "# Leere Datei\n"
                
                if len(new_content) > self.config.MAX_BLOB_SIZE:
                    raise ValueError(
                        f"Datei zu groß für direkten Commit: {file_path} "
                        f"({len(new_content)} bytes > {self.config.MAX_BLOB_SIZE} bytes). "
                        "Bitte Git LFS verwenden."
                    )
                
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
                
                action_symbol = "✨ Erstelle" if action == "create" else "✏️ Ändere"
                logger.info(f"{action_symbol}: {file_path}")
            
            # Prüfe auf tatsächliche Änderungen
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
        
        except ValueError:
            logger.error("Validierungsfehler bei Commit-Erstellung", exc_info=True)
            raise
        
        except GithubException as e:
            logger.error(f"❌ GitHub API Fehler beim Commit: {e}", exc_info=True)
            error_msg = str(e).lower()
            if "rate limit" in error_msg:
                raise Exception("GitHub Rate Limit überschritten. Bitte versuche es später erneut.")
            elif "not found" in error_msg:
                raise Exception(f"Branch oder Repository nicht gefunden: {branch}")
            else:
                raise Exception(f"Commit-Erstellung fehlgeschlagen: {str(e)}")
        
        except Exception as e:
            logger.error(f"❌ Unerwarteter Fehler beim Commit: {e}", exc_info=True)
            raise Exception(f"Commit-Erstellung fehlgeschlagen: {str(e)}")
    
    def create_atomic_commit_with_lock(
        self,
        repo: Repository,
        changes: List[Dict[str, Any]],
        branch: str,
        message: str
    ) -> str:
        """Erstellt atomaren Commit mit Optimistic Locking zur Behandlung konkurrierender Änderungen"""
        
        if not changes:
            raise ValueError("Keine Änderungen bereitgestellt")
        
        if len(message) > self.config.MAX_COMMIT_MSG_LENGTH:
            raise ValueError(f"Commit-Nachricht zu lang (max. {self.config.MAX_COMMIT_MSG_LENGTH} Zeichen)")
        
        if not self.config.ALLOWED_BRANCH_PATTERN.match(branch):
            raise ValueError(f"Ungültiger Branch-Name: {branch}")
        
        try:
            repo.get_branch(branch)
        except GithubException:
            raise ValueError(f"Branch existiert nicht: {branch}")
        except Exception as e:
            logger.error(f"Fehler beim Überprüfen des Branches: {e}")
            raise
        
        max_retries = 3
        base_delay = 2  # Basis für exponentielles Backoff
        
        for attempt in range(max_retries):
            try:
                logger.info(f"🔒 Erstelle atomaren Commit mit Lock auf {repo.full_name}:{branch} (Versuch {attempt + 1}/{max_retries})")
                
                # 1. Hole aktuellen Commit SHA für Optimistic Locking
                base_commit = self.execute_with_rate_limit(
                    repo.get_commit, 
                    f"heads/{branch}"
                )
                expected_sha = base_commit.sha
                base_tree = base_commit.commit.tree
                logger.debug(f"Erwartete Basis-SHA: {expected_sha[:7]}")
                
                # 2. Baue neuen Tree
                input_tree = []
                processed_files = set()
                
                for change in changes:
                    action = change.get("action")
                    file_path = change.get("file_path")
                    
                    if action not in self.config.ALLOWED_ACTIONS:
                        raise ValueError(f"Ungültige Aktion '{action}'. Erlaubt: {self.config.ALLOWED_ACTIONS}")
                    
                    if not file_path or not file_path.strip():
                        logger.warning(f"Überspringe ungültigen Datei-Pfad: {file_path}")
                        continue
                    
                    file_path = file_path.strip("/")
                    
                    if file_path in processed_files:
                        logger.warning(f"Überspringe duplikate Datei: {file_path}")
                        continue
                    
                    processed_files.add(file_path)
                    
                    if action == "delete":
                        logger.info(f"🗑️ Lösche Datei: {file_path}")
                        continue
                    
                    new_content = change.get("new_content", "")
                    
                    if not new_content and action == "create":
                        logger.warning(f"⚠️ Leerer Inhalt für neue Datei: {file_path}")
                        new_content = "# Leere Datei\n"
                    
                    if len(new_content) > self.config.MAX_BLOB_SIZE:
                        raise ValueError(
                            f"Datei zu groß für direkten Commit: {file_path} "
                            f"({len(new_content)} bytes > {self.config.MAX_BLOB_SIZE} bytes). "
                            "Bitte Git LFS verwenden."
                        )
                    
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
                    
                    action_symbol = "✨ Erstelle" if action == "create" else "✏️ Ändere"
                    logger.info(f"{action_symbol}: {file_path}")
                
                # Prüfe auf tatsächliche Änderungen
                if not input_tree and not any(c.get("action") == "delete" for c in changes):
                    logger.warning("Keine Änderungen zum Committen")
                    return expected_sha
                
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
                
                # 5. Hole aktuelle Referenz und prüfe auf konkurrierende Änderungen
                ref = self.execute_with_rate_limit(repo.get_git_ref, f"heads/{branch}")
                
                # Frühe Erkennung von konkurrierenden Modifikationen
                if ref.object.sha != expected_sha:
                    if attempt < max_retries - 1:
                        delay = base_delay ** attempt
                        logger.warning(
                            f"⚠️ Konkurrierende Modifikation erkannt (erwartet: {expected_sha[:7]}, "
                            f"aktuell: {ref.object.sha[:7]}). Warte {delay}s und versuche erneut..."
                        )
                        time.sleep(delay)
                        continue
                    else:
                        raise Exception("Commit fehlgeschlagen: Zu viele konkurrierende Modifikationen")
                
                # 6. Update Branch Ref mit force=False (sichert CAS-Verhalten)
                try:
                    self.execute_with_rate_limit(ref.edit, commit.sha, force=False)
                    logger.info(f"🚀 Branch {branch} aktualisiert auf {commit.sha[:7]}")
                    return commit.sha
                except GithubException as e:
                    error_msg = str(e).lower()
                    if "not a fast forward" in error_msg or "reference already exists" in error_msg:
                        if attempt < max_retries - 1:
                            delay = base_delay ** attempt
                            logger.warning(
                                f"⚠️ Fast-forward fehlgeschlagen (konkurrierende Änderung während Update). "
                                f"Warte {delay}s und versuche erneut..."
                            )
                            time.sleep(delay)
                            continue
                        else:
                            raise Exception("Commit fehlgeschlagen: Zu viele konkurrierende Modifikationen")
                    raise
                
            except ValueError:
                logger.error("Validierungsfehler bei Commit-Erstellung", exc_info=True)
                raise
            
            except GithubException as e:
                logger.error(f"❌ GitHub API Fehler beim Commit: {e}", exc_info=True)
                error_msg = str(e).lower()
                if "rate limit" in error_msg:
                    raise Exception("GitHub Rate Limit überschritten. Bitte versuche es später erneut.")
                elif "not found" in error_msg:
                    raise Exception(f"Branch oder Repository nicht gefunden: {branch}")
                else:
                    raise Exception(f"Commit-Erstellung fehlgeschlagen: {str(e)}")
            
            except Exception as e:
                logger.error(f"❌ Unerwarteter Fehler beim Commit: {e}", exc_info=True)
                raise Exception(f"Commit-Erstellung fehlgeschlagen: {str(e)}")
        
        raise Exception("Unerwartet: Max retries ohne Ausnahme")
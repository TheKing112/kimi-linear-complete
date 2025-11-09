import os
import aiohttp
from pathlib import Path
from typing import List, Dict, Any
import re
import logging

logger = logging.getLogger("repo-analyzer")

class RepositoryAnalyzer:
    def __init__(self):
        self.IGNORE_PATHS = {
            ".git", "__pycache__", ".venv", "venv", "node_modules",
            "dist", "build", ".idea", ".vscode", "*.pyc", "*.pyo"
        }
        self.SUPPORTED_EXTENSIONS = {".py", ".js", ".ts", ".tsx", ".jsx", ".md", ".txt"}
    
    async def clone_and_analyze(self, repo_url: str, branch: str, user_id: str, auto_analyze: bool):
        """Klone und analysiere Repository asynchron"""
        from .github_client import GitHubClient
        
        github = GitHubClient()
        repo_dir = github.clone_repo(repo_url, branch)
        
        if auto_analyze:
            await self.analyze_repository(repo_dir, user_id, repo_url)
    
    async def analyze_repository(self, repo_path: str, user_id: str, repo_url: str):
        """Analysiere alle Dateien und speichere in Cognee"""
        logger.info(f"Analysiere Repository: {repo_path}")
        
        files_processed = 0
        total_size = 0
        
        for file_path in Path(repo_path).rglob("*"):
            if self.should_ignore(file_path):
                continue
            
            if file_path.suffix not in self.SUPPORTED_EXTENSIONS:
                continue
            
            try:
                file_content = file_path.read_text(encoding="utf-8")
                relative_path = file_path.relative_to(repo_path)
                
                # Speichere in Cognee Memory
                await self.store_file_in_cognee(
                    user_id=user_id,
                    content=file_content,
                    file_path=str(relative_path),
                    repo_url=repo_url
                )
                
                files_processed += 1
                total_size += len(file_content)
                
                if files_processed % 10 == 0:
                    logger.info(f"  → {files_processed} Dateien verarbeitet...")
            
            except Exception as e:
                logger.warning(f"Fehler bei {file_path}: {e}")
        
        logger.info(f"✅ Repository-Analyse abgeschlossen: {files_processed} Dateien, {total_size/1024:.2f} KB")
    
    def should_ignore(self, path: Path) -> bool:
        """Prüfe ob Pfad ignoriert werden soll"""
        for ignore in self.IGNORE_PATHS:
            if ignore in str(path):
                return True
        return False
    
    async def store_file_in_cognee(self, user_id: str, content: str, file_path: str, repo_url: str):
        """Speichere Datei in Cognee über API"""
        cognee_url = os.getenv("COGNEE_URL", "http://cognee:8001")
        
        metadata = {
            "type": "repository_file",
            "file_path": file_path,
            "repo_url": repo_url,
            "language": self.detect_language(file_path),
            "file_size": len(content),
            "lines": len(content.splitlines())
        }
        
        payload = {
            "user_id": user_id,
            "content": content,
            "metadata": metadata
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{cognee_url}/memory/store",
                json=payload
            ) as resp:
                if resp.status != 201:
                    logger.warning(f"Cognee Store fehlgeschlagen: {resp.status}")
    
    def detect_language(self, file_path: str) -> str:
        """Erkenne Programmiersprache anhand Extension"""
        ext_map = {
            ".py": "python", ".js": "javascript", ".ts": "typescript",
            ".tsx": "typescript", ".jsx": "javascript", ".md": "markdown"
        }
        return ext_map.get(Path(file_path).suffix, "text")
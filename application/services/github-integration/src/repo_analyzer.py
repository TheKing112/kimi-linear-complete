import os
import aiohttp
import aiofiles
import asyncio
from pathlib import Path
from typing import List, Dict, Any
import re
import logging

logger = logging.getLogger("repo-analyzer")

class RepositoryAnalyzer:
    def __init__(self):
        self.IGNORE_PATHS = {
            ".git", "__pycache__", ".venv", "venv", "node_modules",
            "dist", "build", ".idea", ".vscode", ".pytest_cache",
            ".mypy_cache", ".tox", "htmlcov", ".coverage",
            "*.pyc", "*.pyo", "*.pyd", ".DS_Store"
        }
        self.IGNORE_PATTERNS = [
            r".*\.egg-info$",
            r".*\.so$",
            r".*\.dylib$",
            r".*\.log$",
            r".*\.tmp$"
        ]
        self.SUPPORTED_EXTENSIONS = {
            ".py", ".js", ".ts", ".tsx", ".jsx",
            ".md", ".txt", ".json", ".yaml", ".yml",
            ".toml", ".ini", ".cfg"
        }
        self.MAX_FILE_SIZE = 1_000_000  # 1MB
    
    async def clone_and_analyze(self, repo_url: str, branch: str, user_id: str, auto_analyze: bool):
        """Klone und analysiere Repository asynchron"""
        from .github_client import GitHubClient
        
        github = GitHubClient()
        repo_dir = github.clone_repo(repo_url, branch)
        
        if auto_analyze:
            await self.analyze_repository(repo_dir, user_id, repo_url)
    
    async def analyze_repository(self, repo_path: str, user_id: str, repo_url: str):
        """Analyze repository with async file operations"""
        logger.info(f"Analyzing repository: {repo_path}")
        
        files_processed = 0
        total_size = 0
        tasks = []
        
        for file_path in Path(repo_path).rglob("*"):
            if self.should_ignore(file_path):
                continue
            
            if file_path.suffix not in self.SUPPORTED_EXTENSIONS:
                continue
            
            if not file_path.is_file():
                continue
            
            task = self.process_file(file_path, repo_path, user_id, repo_url)
            tasks.append(task)
            
            if len(tasks) >= 10:
                results = await asyncio.gather(*tasks, return_exceptions=True)
                for result in results:
                    if isinstance(result, Exception):
                        logger.warning(f"File processing error: {result}")
                    elif result:
                        files_processed += 1
                        total_size += result
                
                tasks = []
                
                if files_processed % 50 == 0:
                    logger.info(f"  → {files_processed} files processed...")
        
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for result in results:
                if isinstance(result, Exception):
                    logger.warning(f"File processing error: {result}")
                elif result:
                    files_processed += 1
                    total_size += result
        
        logger.info(
            f"✅ Repository analysis complete: {files_processed} files, "
            f"{total_size/1024:.2f} KB"
        )
    
    async def process_file(
        self,
        file_path: Path,
        repo_path: str,
        user_id: str,
        repo_url: str
    ) -> int:
        """Process single file"""
        try:
            file_content = await self.read_file_async(file_path)
            relative_path = file_path.relative_to(repo_path)
            
            if not file_content.strip():
                return 0
            
            if len(file_content) > self.MAX_FILE_SIZE:
                logger.warning(f"Skipping large file: {relative_path}")
                return 0
            
            await self.store_file_in_cognee(
                user_id=user_id,
                content=file_content,
                file_path=str(relative_path),
                repo_url=repo_url
            )
            
            return len(file_content)
        
        except Exception as e:
            logger.warning(f"Error processing {file_path}: {e}")
            return 0
    
    async def read_file_async(self, file_path: Path) -> str:
        """Read file asynchronously"""
        try:
            async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
                return await f.read()
        except UnicodeDecodeError:
            async with aiofiles.open(file_path, 'r', encoding='latin-1') as f:
                return await f.read()
        except Exception as e:
            logger.warning(f"Could not read {file_path}: {e}")
            return ""
    
    def should_ignore(self, path: Path) -> bool:
        """Check if path should be ignored"""
        path_str = str(path)
        
        for ignore in self.IGNORE_PATHS:
            if ignore in path_str:
                return True
        
        for pattern in self.IGNORE_PATTERNS:
            if re.match(pattern, path_str):
                return True
        
        try:
            if path.is_file() and path.stat().st_size > self.MAX_FILE_SIZE:
                logger.info(f"Skipping large file: {path}")
                return True
        except:
            pass
        
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
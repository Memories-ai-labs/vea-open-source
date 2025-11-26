"""
Local filesystem storage adapter - same interface as GoogleCloudStorage.

This allows the codebase to work with local files using the same API
as GCS, enabling easy switching between local and cloud storage.
"""

import os
import shutil
import logging
from pathlib import Path
from typing import List

logger = logging.getLogger(__name__)


class LocalStorage:
    """
    Local filesystem storage with the same interface as GoogleCloudStorage.

    The 'bucket' parameter maps to a base directory on disk.
    All paths are relative to this base directory.
    """

    def __init__(self, base_path: str = "data"):
        """
        Initialize local storage.

        Args:
            base_path: Base directory for all storage operations (default: "data")
        """
        self.base_path = Path(base_path).resolve()
        self.base_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"LocalStorage initialized with base path: {self.base_path}")

    def _resolve_path(self, bucket: str, path: str) -> Path:
        """Resolve a bucket/path to an absolute local path."""
        # In local mode, bucket is ignored (or could be a subdirectory)
        # We use the path directly relative to base_path
        return self.base_path / path.lstrip("/")

    def path_exists(self, bucket: str, path: str) -> bool:
        """
        Check if a file or folder exists at the given path.

        Args:
            bucket: Ignored in local mode (kept for API compatibility)
            path: Relative path to check

        Returns:
            bool: True if path exists
        """
        local_path = self._resolve_path(bucket, path)
        return local_path.exists()

    def all_files_exist(self, bucket: str, base_path: str, filenames: List[str]) -> bool:
        """
        Check if all given files exist in the specified path.

        Args:
            bucket: Ignored in local mode
            base_path: Base folder path
            filenames: List of filenames to check

        Returns:
            bool: True if all files exist
        """
        for fname in filenames:
            full_path = os.path.join(base_path, fname)
            if not self.path_exists(bucket, full_path):
                return False
        return True

    def upload_files(self, bucket_name: str, local_path: str, dest_path: str):
        """
        Copy files from source to destination within local storage.

        In local mode, this copies files from local_path to base_path/dest_path.

        Args:
            bucket_name: Ignored in local mode
            local_path: Source path (can be file or directory)
            dest_path: Destination path relative to base_path
        """
        source = Path(local_path)
        dest = self._resolve_path(bucket_name, dest_path)

        if source.is_file():
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, dest)
            logger.info(f"Copied file {source} to {dest}")
        elif source.is_dir():
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(source, dest)
            logger.info(f"Copied directory {source} to {dest}")
        else:
            raise FileNotFoundError(f"Source path not found: {local_path}")

    def download_files(self, bucket_name: str, src_path: str, local_path: str):
        """
        Copy files from storage to a local destination.

        In local mode, this copies from base_path/src_path to local_path.

        Args:
            bucket_name: Ignored in local mode
            src_path: Source path relative to base_path
            local_path: Destination path
        """
        source = self._resolve_path(bucket_name, src_path)
        dest = Path(local_path)

        if src_path.endswith("/"):
            # Directory download
            if not source.exists():
                raise FileNotFoundError(f"Source directory not found: {source}")
            dest.mkdir(parents=True, exist_ok=True)
            for item in source.iterdir():
                if item.is_file():
                    shutil.copy2(item, dest / item.name)
                    logger.info(f"Copied {item} to {dest / item.name}")
        else:
            # File download
            if not source.exists():
                raise FileNotFoundError(f"Source file not found: {source}")
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, dest)
            logger.info(f"Copied {source} to {dest}")

    def delete_blob(self, bucket: str, path: str):
        """
        Delete a file.

        Args:
            bucket: Ignored in local mode
            path: Path to file to delete
        """
        local_path = self._resolve_path(bucket, path)
        if local_path.exists() and local_path.is_file():
            local_path.unlink()
            logger.info(f"Deleted file: {local_path}")

    def delete_folder(self, bucket: str, path: str):
        """
        Delete a folder and all its contents.

        Args:
            bucket: Ignored in local mode
            path: Path to folder to delete
        """
        local_path = self._resolve_path(bucket, path)
        if local_path.exists():
            shutil.rmtree(local_path)
            logger.info(f"Deleted folder: {local_path}")

    def list_folder(self, bucket: str, path: str) -> List[tuple]:
        """
        List contents of a folder.

        Args:
            bucket: Ignored in local mode
            path: Path to folder

        Returns:
            List of (name, full_path) tuples
        """
        local_path = self._resolve_path(bucket, path)
        if not local_path.exists():
            return []

        results = []
        for item in local_path.iterdir():
            name = item.stem  # filename without extension
            rel_path = str(item.relative_to(self.base_path))
            results.append((name, rel_path))
        return results

    def list_blobs(self, bucket: str, match_glob: str = None, max_result: int = None) -> List[str]:
        """
        List all files matching a pattern.

        Args:
            bucket: Ignored in local mode
            match_glob: Glob pattern to match
            max_result: Maximum number of results

        Returns:
            List of relative paths
        """
        if match_glob:
            files = list(self.base_path.glob(match_glob))
        else:
            files = list(self.base_path.rglob("*"))

        results = [str(f.relative_to(self.base_path)) for f in files if f.is_file()]

        if max_result:
            results = results[:max_result]

        return results

    def get_public_download_url(self, bucket_name: str, path: str, expired_in_hour: int = 1) -> str:
        """
        Get a URL for the file. In local mode, returns the file path.

        Args:
            bucket_name: Ignored in local mode
            path: Path to file
            expired_in_hour: Ignored in local mode

        Returns:
            Local file path as string
        """
        local_path = self._resolve_path(bucket_name, path)
        return str(local_path)

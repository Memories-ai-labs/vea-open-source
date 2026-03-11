"""
Storage factory - returns appropriate storage client based on configuration.
"""

from src.config import is_local_mode


def get_storage_client():
    """
    Get the appropriate storage client based on configuration.

    Returns:
        LocalStorage (open-source version always uses local storage).
    """
    from lib.oss.local_storage import LocalStorage
    return LocalStorage(base_path=".")

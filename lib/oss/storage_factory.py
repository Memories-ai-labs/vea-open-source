"""
Storage factory - returns appropriate storage client based on configuration.
"""

from src.config import is_local_mode, CREDENTIAL_PATH


def get_storage_client():
    """
    Get the appropriate storage client based on configuration.

    Returns:
        LocalStorage if in local mode, GoogleCloudStorage if in cloud mode.
    """
    if is_local_mode():
        from lib.oss.local_storage import LocalStorage
        return LocalStorage()
    else:
        from lib.oss.gcp_oss import GoogleCloudStorage
        from lib.oss.auth import credentials_from_file
        return GoogleCloudStorage(credentials=credentials_from_file(CREDENTIAL_PATH))

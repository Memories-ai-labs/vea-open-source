from google.auth.credentials import Credentials
from google.oauth2 import service_account


def credentials_from_file(filepath: str) -> Credentials | Exception:
    return service_account.Credentials.from_service_account_file(filepath)
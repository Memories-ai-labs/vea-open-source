from datetime import timedelta
from typing import List, Sequence
import logging
import os
from google.api_core.exceptions import NotFound
from google.cloud import storage
from google.cloud.storage import transfer_manager
from google.cloud.storage.transfer_manager import THREAD
from tqdm import tqdm

from lib.oss.oss import OSSDataMapping, OSSPathMapping
from tenacity import retry, stop_after_attempt, wait_exponential

# Set up logging
logger = logging.getLogger(__name__)

class GoogleCloudStorage:
    def __init__(self, credentials=None) -> None:
        super().__init__()
        self.client = storage.Client(credentials=credentials)
        self.__worker_type = THREAD

    @classmethod
    def parse_uri(cls, uri: str) -> tuple[str, str]:
        if uri.startswith("gs://"):
            uri = uri[5:]
            uri = uri.strip("/")
            bucket, blob = uri.split("/", 1)
            return bucket, blob
        else:
            raise ValueError(f"Invalid uri: {uri}")


    def path_exists(self, bucket: str, gcs_path: str) -> bool:
        """
        Checks whether a file or folder exists in GCS at the given path.
        Works for both exact blob paths (files) and folder prefixes.

        Args:
            bucket (str): The name of the GCS bucket.
            gcs_path (str): The full GCS path (e.g., 'folder/file.txt' or 'folder/').

        Returns:
            bool: True if the path exists as a file or folder, False otherwise.
        """
        bucket = self.client.bucket(bucket)
        # Check for exact match (file)
        blob = bucket.blob(gcs_path)
        if blob.exists():
            return True
        # Check for folder existence
        blobs = list(bucket.list_blobs(prefix=gcs_path))
        return len(blobs) > 0

    def all_files_exist(self, bucket: str, base_path: str, filenames: List[str]) -> bool:
        """
        Check if all given files exist in the specified GCS bucket and path.

        Args:
            bucket (str): GCS bucket name.
            base_path (str): Folder/prefix in GCS (e.g., 'indexing/movie_name/').
            filenames (List[str]): List of filenames (e.g., ['plot.txt', 'scenes.json'])

        Returns:
            bool: True if all files exist, False otherwise.
        """
        for fname in filenames:
            full_path = os.path.join(base_path, fname).replace("\\", "/")
            if not self.path_exists(bucket, full_path):
                return False
        return True

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10), reraise=True)
    def upload_files(self, bucket_name: str, local_path: str, gcs_path: str):
        bucket = self.client.bucket(bucket_name)

        if os.path.isfile(local_path):
            blob = bucket.blob(gcs_path)
            blob.upload_from_filename(local_path)
            logger.info(f"Uploaded file {local_path} to gs://{bucket_name}/{gcs_path}")
        elif os.path.isdir(local_path):
            for root, _, files in os.walk(local_path):
                for file in files:
                    full_path = os.path.join(root, file)
                    rel_path = os.path.relpath(full_path, local_path)
                    blob_path = os.path.join(gcs_path, rel_path).replace("\\", "/")
                    blob = bucket.blob(blob_path)
                    blob.upload_from_filename(full_path)
                    logger.info(f"Uploaded {full_path} to gs://{bucket_name}/{blob_path}")

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10), reraise=True)
    def download_files(self, bucket_name: str, gcs_path: str, local_path: str):
        bucket = self.client.bucket(bucket_name)

        if gcs_path.endswith("/"):
            blobs = bucket.list_blobs(prefix=gcs_path)
            for blob in blobs:
                # Skip the "folder" blob itself if it exists (e.g., 'folder/')
                if blob.name.endswith("/") or blob.name == gcs_path:
                    continue

                rel_path = os.path.relpath(blob.name, gcs_path)
                local_file_path = os.path.join(local_path, rel_path)

                os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
                blob.download_to_filename(local_file_path)
                logger.info(f"Downloaded gs://{bucket_name}/{blob.name} to {local_file_path}")
        else:
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            blob = bucket.blob(gcs_path)
            blob.download_to_filename(local_path)
            logger.info(f"Downloaded gs://{bucket_name}/{gcs_path} to {local_path}")

    def delete_blob(self, bucket, blob_path: str):
        """Deletes a blob from the bucket."""
        bucket = self.client.bucket(bucket)
        blob = bucket.blob(blob_path)
        try:
            blob.delete()
        except NotFound:
            return None

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        reraise=True
    )
    def delete_folder(self, bucket, blob_path: str) -> None:
        bucket = self.client.bucket(bucket)
        blobs = bucket.list_blobs(prefix=blob_path)
        try:
            for blob in blobs:
                blob.delete()
        except Exception as e:
            raise e
    def get_public_download_url(
            self, bucket_name: str, blob_path: str, expired_in_hour=1
        ) -> str:
            bucket = self.client.bucket(bucket_name)
            blob = bucket.blob(blob_path)
            expiration = timedelta(hours=expired_in_hour)
            signed_url = blob.generate_signed_url(expiration=expiration)
            return signed_url

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        reraise=True
    )
    def list_folder(
            self, bucket, blob_path: str
        ) -> List[str]:
            bucket = self.client.bucket(bucket)
            blobs = bucket.list_blobs(prefix=blob_path)
            return [(os.path.basename(blob.name).split('.')[0], blob.name) for blob in blobs if blob.name != blob_path]

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        reraise=True
    )
    def list_blobs(
            self, bucket, match_glob: str | None = None, max_result: int | None = None
        ) -> List[str]:
            """Lists all the blobs in the bucket that begin with the prefix.

            This can be used to list all blobs in a "folder", e.g. "public/".

            The delimiter argument can be used to restrict the results to only the
            "files" in the given "folder". Without the delimiter, the entire tree under
            the prefix is returned. For example, given these blobs:

                a/1.txt
                a/b/2.txt

            If you specify prefix ='a/', without a delimiter, you'll get back:

                a/1.txt
                a/b/2.txt

            However, if you specify prefix='a/' and delimiter='/', you'll get back
            only the file directly under 'a/':

                a/1.txt

            As part of the response, you'll also get back a blobs.prefixes entity
            that lists the "subfolders" under `a/`:

                a/b/
            """

            storage_client = storage.Client()

            # Note: Client.list_blobs requires at least package version 1.17.0.
            blobs = storage_client.list_blobs(
                bucket, match_glob=match_glob, max_results=max_result
            )

            return [blob.name for blob in blobs]



from datetime import timedelta
from typing import List, Sequence
import logging
import os
from google.api_core.exceptions import NotFound
from google.cloud import storage
from google.cloud.storage import transfer_manager
from google.cloud.storage.transfer_manager import THREAD
from tqdm import tqdm

from src.utils.oss.oss import OSSDataMapping, OSSPathMapping
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

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        reraise=True
    )
    def upload_from_file(self, filepath: str, bucket_name: str, blob_path: str) -> None:
        bucket = self.client.bucket(bucket_name)
        blob = bucket.blob(blob_path)
        blob.upload_from_filename(filepath)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        reraise=True
    )
    def download_to_memory(
        self,
        bucket_name: str,
        blob_path: str,
    ) -> bytes:
        bucket = self.client.bucket(bucket_name)
        blob = bucket.blob(blob_path)
        return blob.download_as_bytes()

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        reraise=True
    )
    def download_to_file(self, bucket_name, blob_path, local_path) -> None:
        bucket = self.client.bucket(bucket_name)
        blob = bucket.blob(blob_path)
        blob.download_to_filename(local_path)
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        reraise=True
    )
    async def download_to_file_with_progress(self, bucket_name, blob_path, local_path) -> None:
        bucket = self.client.bucket(bucket_name)
        blob = bucket.blob(blob_path)
        
        # Get the total size of the file
        blob.reload()
        
        with open(local_path, 'wb') as f:
            with tqdm.wrapattr(f, "write", total=blob.size) as file_obj:
                # blob.download_to_file is deprecated
                self.client.download_blob_to_file(blob, file_obj)
        logger.info(f"Downloaded {blob_path} to {local_path}")


    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        reraise=True
    )
    def upload_from_memory(
        self, bucket_name: str, blob_path: str, data, content_type="text/plain"
    ) -> None:
        bucket = self.client.bucket(bucket_name)
        blob = bucket.blob(blob_path)
        blob.upload_from_string(data, content_type)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        reraise=True
    )
    def upload_batch_from_memory(
        self, bucket_name, blob_map: Sequence[OSSDataMapping]
    ) -> None:
        bucket = self.client.bucket(bucket_name)
        new_blob_map = [(data, bucket.blob(blob)) for data, blob in blob_map]
        results = transfer_manager.upload_many(
            new_blob_map, worker_type=self.__worker_type, max_workers=8
        )
        for name, result in zip(new_blob_map, results):
            if isinstance(result, Exception):
                logger.error(f"Failed to upload {name} due to exception: {result}")
            else:
                logger.info(f"Uploaded {name} to {bucket.name}.")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        reraise=True
    )
    def upload_batch_from_file(
        self, bucket_name, blob_map: Sequence[OSSPathMapping]
    ) -> None:
        bucket = self.client.bucket(bucket_name)
        new_blob_map = [(path, bucket.blob(blob)) for path, blob in blob_map]
        results = transfer_manager.upload_many(
            new_blob_map, worker_type=self.__worker_type, max_workers=8
        )
        for name, result in zip(new_blob_map, results):
            if isinstance(result, Exception):
                logger.error(f"Failed to upload {name} due to exception: {result}")
            else:
                logger.info(f"Uploaded {name} to {bucket.name}.")

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

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        reraise=True
    )
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


if __name__ == "__main__":
    from src.utils.oss.auth import credentials_from_file

    bucket = "xvu-datasets"

    credentail = credentials_from_file(
        "E:/OpenInterX Code Source/vea-playground/backend/config/gen-lang-client-0057517563-0319d78ed5fe.json"
    )

    gcp = GoogleCloudStorage(credentail)

    def __test_list_blob():
        print(len(gcp.list_blobs(bucket)))

    __test_list_blob()
import io
from abc import ABC, abstractmethod
from typing import Sequence, Tuple

OSSPathMapping = Tuple[str, str]
OSSDataMapping = Tuple[io.IOBase, str]


class OSSClient(ABC):
    @abstractmethod
    def upload_from_file(self, bucket, src, blob_path) -> None | Exception:
        raise NotImplementedError

    @abstractmethod
    def upload_from_memory(self, bucket, data, blob_path) -> None | Exception:
        raise NotImplementedError

    @abstractmethod
    def upload_batch_from_file(self, bucket: str, blob_map: Sequence[OSSPathMapping]) -> None | Exception:
        pass

    @abstractmethod
    def upload_batch_from_memory(self, bucket, blob_map: Sequence[OSSDataMapping]) -> None | Exception:
        pass
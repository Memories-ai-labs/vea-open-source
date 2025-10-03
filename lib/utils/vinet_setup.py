"""Utility script to clone ViNet repository and download associated assets."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import tarfile
from pathlib import Path
from typing import Optional

import requests
import re

REPO_URL = "https://github.com/ViNet-Saliency/vinet_v2.git"
FILE_ID = "12UeAsdiD2xPLmoLRDcE_HjAUjxFdmw5N"
GOOGLE_DRIVE_URL = "https://docs.google.com/uc?export=download"


def project_root() -> Path:
    """Return absolute path to the project root directory."""
    return Path(__file__).resolve().parents[2]


def clone_repo(destination: Path, force: bool = False) -> None:
    """Clone the ViNet repository into the destination directory."""
    repo_dir = destination / "vinet_v2"
    if repo_dir.exists():
        if not force:
            print(f"[INFO] Repository already exists at {repo_dir}. Skipping clone.")
            return
        print(f"[INFO] Removing existing repository at {repo_dir} due to --force.")
        shutil.rmtree(repo_dir)

    print(f"[INFO] Cloning {REPO_URL} into {destination}...")
    subprocess.run(["git", "clone", REPO_URL], cwd=str(destination), check=True)
    print("[SUCCESS] Repository cloned.")


def get_confirm_token(response: requests.Response) -> Optional[str]:
    """Extract confirmation token for large Google Drive downloads."""
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            return value
    return None


def _parse_confirmation_form(html: str):
    """Parse confirm token, uuid, and form action from Google Drive HTML."""
    confirm_match = re.search(r'name="confirm"\s+value="([^"]+)"', html)
    uuid_match = re.search(r'name="uuid"\s+value="([^"]+)"', html)
    action_match = re.search(r'<form[^>]*id="download-form"[^>]*action="([^"]+)"', html)
    confirm = confirm_match.group(1) if confirm_match else None
    uuid_token = uuid_match.group(1) if uuid_match else None
    action_url = action_match.group(1) if action_match else None
    return confirm, uuid_token, action_url


def _resolve_drive_response(session: requests.Session, response: requests.Response, file_id: str) -> requests.Response:
    """Handle virus-scan warning pages and confirmation tokens."""
    while True:
        token = get_confirm_token(response)
        if token:
            response.close()
            params = {"id": file_id, "export": "download", "confirm": token}
            response = session.get(GOOGLE_DRIVE_URL, params=params, stream=True)
            continue

        content_type = response.headers.get("content-type", "").lower()
        if "text/html" in content_type:
            html = response.content.decode("utf-8", errors="ignore")
            confirm, uuid_token, action_url = _parse_confirmation_form(html)
            if not confirm and "download anyway" in html.lower():
                confirm = "t"
            if not confirm:
                raise RuntimeError("Unable to extract confirmation token from Google Drive response.")

            params = {"id": file_id, "export": "download", "confirm": confirm}
            if uuid_token:
                params["uuid"] = uuid_token
            download_url = action_url or "https://drive.usercontent.google.com/download"
            response.close()
            response = session.get(download_url, params=params, stream=True)
            continue

        return response


def download_google_drive_file(file_id: str, destination: Path, chunk_size: int = 32768) -> None:
    """Download a file from Google Drive handling large-file confirmation tokens."""
    session = requests.Session()
    response = session.get(GOOGLE_DRIVE_URL, params={"id": file_id, "export": "download"}, stream=True)
    response = _resolve_drive_response(session, response, file_id)
    response.raise_for_status()
    with destination.open("wb") as f:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
    print(f"[SUCCESS] Downloaded file to {destination}.")


def _safe_tar_members(tar: tarfile.TarFile, destination: Path):
    """Yield members that will stay within the destination directory."""
    dest_path = destination.resolve()
    for member in tar.getmembers():
        member_path = (dest_path / member.name).resolve()
        if not member_path.is_relative_to(dest_path):
            raise RuntimeError(f"Unsafe path detected in tar archive: {member.name}")
        yield member


def extract_tarball(archive_path: Path, destination: Path) -> None:
    """Extract a tar.gz archive into the destination directory."""
    print(f"[INFO] Extracting {archive_path} into {destination}...")
    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(path=destination, members=_safe_tar_members(tar, destination))
    print("[SUCCESS] Extraction complete.")


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Clone ViNet repo and download assets.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-clone repository and re-download archive even if they exist.",
    )
    parser.add_argument(
        "--archive-path",
        type=Path,
        default=None,
        help="Optional custom output path for the downloaded tar.gz archive.",
    )
    args = parser.parse_args(argv)

    root = project_root()
    clone_repo(root, force=args.force)

    repo_dir = root / "vinet_v2"
    archive_path = args.archive_path or (repo_dir / "vinet_v2_assets.tar.gz")
    if archive_path.exists() and not args.force:
        print(f"[INFO] Archive already exists at {archive_path}. Skipping download.")
    else:
        if archive_path.exists():
            print(f"[INFO] Removing existing archive at {archive_path} due to --force.")
            archive_path.unlink()
        print(f"[INFO] Downloading asset archive to {archive_path}...")
        download_google_drive_file(FILE_ID, archive_path)

    extract_tarball(archive_path, repo_dir)

    if archive_path.exists():
        archive_path.unlink()
        print(f"[INFO] Removed archive {archive_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Resolve manifest photos locally or from a public dataset ZIP.
Authors: Team 8; implementation assisted by OpenAI Codex.
External sources: Python zipfile/urllib APIs and gdown.download API
(https://github.com/wkentaro/gdown); no implementation copied.
"""
import argparse
import hashlib
import importlib
import json
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
import shutil
import stat
import subprocess
import sys
import tempfile
from urllib.parse import parse_qs, urlparse
from urllib.request import Request, urlopen
import zipfile

IMAGE_KEYS = ("left", "right", "raw_left", "raw_right")
MAX_ARCHIVE_BYTES = 8 * 1024 ** 3
MAX_EXPANDED_BYTES = 20 * 1024 ** 3
RESERVED = {"src", "scripts", "tools", "docs", "build", ".git", ".codex", ".agents"}


def archive_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def relative_path(value):
    """Portable, confined paths for archives and workspace destinations."""
    if not isinstance(value, str) or not value or "\x00" in value:
        raise ValueError("Dataset path must be a nonempty string")
    value = value.replace("\\", "/")
    path = PurePosixPath(value)
    if path.is_absolute() or PureWindowsPath(value).drive or ".." in path.parts:
        raise ValueError(f"Unsafe relative dataset path: {value}")
    if any(":" in part for part in path.parts):
        raise ValueError(f"Unsafe dataset path: {value}")
    return Path(*path.parts)


def references(manifest):
    return sorted({pair[key] for pair in manifest.get("pairs", [])
                   for key in IMAGE_KEYS if pair.get(key)})


def _destination(root, value):
    relative = relative_path(value)
    if (not relative.parts or relative.parts[0].lower() in RESERVED
            or relative.suffix.lower() in (".py", ".ipynb", ".json", ".cpp", ".cu", ".h", ".cuh", ".md", ".txt")):
        raise ValueError(f"Dataset destination conflicts with submitted source: {value}")
    destination = (root / relative).resolve()
    if not destination.is_relative_to(root.resolve()):
        raise ValueError(f"Dataset destination escapes workspace: {value}")
    return destination


def copy_referenced_images(source_manifest, workspace):
    """Copy arbitrary relative photo directories, retaining manifest layout.

    Existing absolute paths (for example mounted private Drive files) remain
    usable without copying. Missing absolute paths cannot be filled from a ZIP.
    """
    source_manifest, workspace = Path(source_manifest).resolve(), Path(workspace).resolve()
    manifest = json.loads(source_manifest.read_text(encoding="utf-8"))
    copied = []
    for value in references(manifest):
        source = (source_manifest.parent / value).resolve()
        if Path(value).is_absolute() or PureWindowsPath(value).drive:
            continue
        destination = _destination(workspace, value)
        if source.is_file() and source != destination:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            copied.append(value)
    return copied


def safe_extract(archive_path, destination):
    """Validate every entry before writing; never extract links or traversals."""
    destination = Path(destination).resolve()
    with zipfile.ZipFile(archive_path) as archive:
        members = archive.infolist()
        if len(members) > 10000 or sum(m.file_size for m in members) > MAX_EXPANDED_BYTES:
            raise ValueError("Dataset ZIP exceeds extraction limits")
        targets = []
        seen = set()
        for member in members:
            path = relative_path(member.filename)
            target = (destination / path).resolve()
            key = str(target).casefold()
            if key in seen or not target.is_relative_to(destination):
                raise ValueError(f"Duplicate or escaping ZIP path: {member.filename}")
            seen.add(key)
            mode = member.external_attr >> 16
            if stat.S_ISLNK(mode) or (stat.S_IFMT(mode) not in (0, stat.S_IFREG, stat.S_IFDIR)):
                raise ValueError(f"Dataset ZIP contains a nonregular entry: {member.filename}")
            if member.flag_bits & 1:
                raise ValueError("Encrypted dataset ZIP is not supported")
            targets.append((member, target))
        for member, target in targets:
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(member) as source, target.open("wb") as output:
                    shutil.copyfileobj(source, output)


def download_archive(url, destination, allow_install=False):
    parsed = urlparse(url)
    if parsed.scheme not in ("https", "http"):
        raise ValueError("dataset_url must be an HTTP(S) public ZIP URL")
    if parsed.hostname in ("drive.google.com", "docs.google.com", "drive.usercontent.google.com"):
        match = re.search(r"/file/d/([A-Za-z0-9_-]+)", parsed.path)
        file_id = match.group(1) if match else parse_qs(parsed.query).get("id", [None])[0]
        if not file_id or not re.fullmatch(r"[A-Za-z0-9_-]+", file_id):
            raise ValueError("Provide a public Google Drive ZIP file link, not a folder/Docs link")
        try:
            gdown = importlib.import_module("gdown")
        except ImportError:
            if not allow_install:
                raise RuntimeError("Drive ZIP requires gdown; install gdown>=5.2,<6 or use --allow-install")
            subprocess.run([sys.executable, "-m", "pip", "install", "gdown>=5.2,<6"], check=True)
            gdown = importlib.import_module("gdown")
        result = gdown.download(id=file_id, output=str(destination), quiet=False, use_cookies=False)
        if not result:
            raise RuntimeError("Drive download failed; share ZIP as Anyone with the link or mount private Drive")
    else:
        with urlopen(Request(url, headers={"User-Agent": "Group08-Dataset/1.0"}), timeout=45) as response:
            if response.headers.get_content_type() == "text/html":
                raise ValueError("dataset_url returned HTML/login page; provide a downloadable ZIP")
            size = 0
            with Path(destination).open("wb") as output:
                while chunk := response.read(1024 * 1024):
                    size += len(chunk)
                    if size > MAX_ARCHIVE_BYTES:
                        raise ValueError("Dataset ZIP exceeds download limit")
                    output.write(chunk)
    if Path(destination).stat().st_size > MAX_ARCHIVE_BYTES or not zipfile.is_zipfile(destination):
        raise ValueError("Downloaded dataset is not a supported ZIP or exceeds download limit")


def resolve_dataset(manifest_path, archive_path=None, allow_install=False):
    """Fill missing referenced photos; retain the exact configured manifest."""
    manifest_path = Path(manifest_path).resolve()
    root = manifest_path.parent
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    missing = [v for v in references(manifest) if not (root / v).is_file()]
    if not missing:
        return {"status": "available", "downloaded": False, "resolved_files": []}
    for value in missing:
        if Path(value).is_absolute() or PureWindowsPath(value).drive:
            raise FileNotFoundError(f"Missing absolute photo {value}; mount Drive/update portable manifest paths")
        _destination(root, value)
    if archive_path is None and not manifest.get("dataset_url"):
        raise FileNotFoundError("Photos are missing and dataset_url is empty; provide a public dataset ZIP")
    work = root / "out" / "dataset_resolver"
    work.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="resolve_", dir=work) as folder:
        temporary = Path(folder)
        archive = Path(archive_path).resolve() if archive_path is not None else temporary / "dataset.zip"
        if archive_path is None:
            download_archive(manifest["dataset_url"], archive, allow_install=allow_install)
        if archive.stat().st_size > MAX_ARCHIVE_BYTES:
            raise ValueError("Dataset ZIP exceeds archive limit")
        expected = manifest.get("dataset_archive_sha256")
        archive_digest = archive_sha256(archive)
        if expected and archive_digest.lower() != expected.lower():
            raise ValueError("Dataset archive SHA256 does not match configured fingerprint")
        staging = temporary / "extracted"
        safe_extract(archive, staging)
        first = relative_path(missing[0])
        candidates = {staging}
        for file in staging.rglob(first.name):
            if file.is_file() and len(file.relative_to(staging).parts) >= len(first.parts):
                candidate = file
                for _ in first.parts:
                    candidate = candidate.parent
                candidates.add(candidate)
        suitable = [base for base in sorted(candidates) if all((base / relative_path(v)).is_file() for v in missing)]
        if len(suitable) != 1:
            raise ValueError("Dataset ZIP must contain one common folder with all paths in the manifest")
        for value in missing:
            destination = _destination(root, value)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(suitable[0] / relative_path(value), destination)
    return {"status": "resolved", "downloaded": archive_path is None,
            "archive_sha256": archive_digest, "resolved_files": missing}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--archive", type=Path, help="Local ZIP, including a mounted private Drive ZIP")
    parser.add_argument("--allow-install", action="store_true", help="Install gdown only if a Drive download needs it")
    args = parser.parse_args()
    print(json.dumps(resolve_dataset(args.manifest, args.archive, args.allow_install), indent=2))


if __name__ == "__main__":
    main()

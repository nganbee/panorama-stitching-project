#!/usr/bin/env python3
"""Dataset preparation. Authors: Group 08 (names/IDs pending); AI-assisted tooling.
External sources: Pillow Image.resize API; no external implementation copied.
"""
import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

from PIL import Image, ImageOps

RESOLUTIONS = {"small": (1280, 720), "medium": (1920, 1080), "large": (3840, 2160)}


def image_identity(path):
    """Fingerprint the bytes that will be read and validate their decoded geometry."""
    path = Path(path).resolve()
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    with Image.open(path) as image:
        size = list(image.size)
        image.verify()
    if min(size) <= 0:
        raise ValueError(f"Invalid image dimensions: {path}")
    if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
        raise ValueError(f"Image changed while being inspected: {path}")
    return {"sha256": digest, "size": size}


def load_manifest(path, require_images=True):
    path = Path(path).resolve()
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise ValueError("Expected manifest schema_version=1")
    sites = manifest.get("sites", [])
    ids = [site["site_id"] for site in sites]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate site_id")
    pair_ids = set()
    for pair in manifest.get("pairs", []):
        pid = pair["pair_id"]
        if not pid or pid in pair_ids:
            raise ValueError("pair_id must be nonempty and unique")
        if any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in pid):
            raise ValueError("pair_id must contain only letters, digits, '_' or '-'")
        pair_ids.add(pid)
        if pair["site_id"] not in ids:
            raise ValueError(f"Unknown site_id for {pid}")
        if pair.get("kind") not in ("standard", "challenging"):
            raise ValueError(f"Invalid kind for {pid}")
        for key in ("left", "right"):
            source = (path.parent / pair[key]).resolve()
            if require_images and not source.is_file():
                raise FileNotFoundError(source)
            if require_images:
                image_identity(source)
    return manifest


def requirements(manifest, base=None):
    """PDF is authoritative: >=3 sites, one standard/site, >=1 challenge total."""
    sites = manifest.get("sites", [])
    pairs = manifest.get("pairs", [])
    standard = {p["site_id"] for p in pairs if p["kind"] == "standard"}
    missing = []
    if len(sites) < 3:
        missing.append("Cần ít nhất 3 địa điểm đã đăng ký")
    for site in sites:
        if site["site_id"] not in standard:
            missing.append(f"Thiếu cặp standard của {site['site_id']}")
        if not site.get("display_name") or not site.get("registration"):
            missing.append(f"Chưa điền tên/đăng ký địa điểm {site['site_id']}")
    if not any(p["kind"] == "challenging" for p in pairs):
        missing.append("Cần ít nhất một cặp challenging")
    if not str(manifest.get("dataset_url", "")).startswith(("https://", "http://")):
        missing.append("Chưa điền link tải dataset ảnh gốc và ảnh đã chuẩn bị")
    for pair in pairs:
        if not pair.get("raw_left") or not pair.get("raw_right"):
            missing.append(f"Chưa khai báo ảnh gốc của {pair['pair_id']}")
        elif base is not None:
            for key in ("raw_left", "raw_right"):
                if not (Path(base) / pair[key]).is_file():
                    missing.append(f"Không tìm thấy ảnh gốc {pair['pair_id']}/{key}: {pair[key]}")
    return missing


def prepare(manifest_path, output_dir):
    manifest_path = Path(manifest_path).resolve()
    output_dir = Path(output_dir).resolve()
    manifest = load_manifest(manifest_path)
    manifest_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    output_dir.mkdir(parents=True, exist_ok=True)
    prepared = {**manifest, "source_manifest": str(manifest_path),
                "input_manifest_sha256": manifest_hash, "prepared_pairs": []}
    for pair in manifest["pairs"]:
        images = []
        originals = {key: image_identity(manifest_path.parent / pair[key]) for key in ("left", "right")}
        for key in ("left", "right"):
            with Image.open(manifest_path.parent / pair[key]) as source:
                images.append(ImageOps.exif_transpose(source).convert("RGB"))
        max_w = max(im.width for im in images)
        max_h = max(im.height for im in images)
        for resolution, (target_w, target_h) in RESOLUTIONS.items():
            # Apply one factor to the entire pair: geometry remains compatible.
            scale = min(target_w / max_w, target_h / max_h)
            entry = {**pair, "resolution": resolution, "target_box": [target_w, target_h],
                     "scale": scale, "original_sizes": [list(im.size) for im in images]}
            entry["input_image_sha256"] = {key: originals[key]["sha256"] for key in ("left", "right")}
            entry["prepared_image_sha256"] = {}
            dest = output_dir / pair["pair_id"] / resolution
            dest.mkdir(parents=True, exist_ok=True)
            for key, im in zip(("left", "right"), images):
                size = (max(1, math.floor(im.width * scale + .5)),
                        max(1, math.floor(im.height * scale + .5)))
                resized = im.resize(size, Image.Resampling.LANCZOS)
                file = dest / f"{key}.png"
                resized.save(file)
                entry[key] = file.relative_to(output_dir).as_posix()
                entry[f"{key}_size"] = list(size)
                entry["prepared_image_sha256"][key] = image_identity(file)["sha256"]
            prepared["prepared_pairs"].append(entry)
        for key in ("left", "right"):
            if image_identity(manifest_path.parent / pair[key])["sha256"] != originals[key]["sha256"]:
                raise ValueError(f"Original image changed during preparation: {pair['pair_id']}/{key}")
    if hashlib.sha256(manifest_path.read_bytes()).hexdigest() != manifest_hash:
        raise ValueError("Dataset manifest changed during preparation; prepare again")
    destination = output_dir / "manifest.prepared.json"
    destination.write_text(json.dumps(prepared, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return destination


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path, default=Path("out/prepared"))
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--strict", action="store_true", help="require all team dataset fields")
    args = parser.parse_args()
    manifest = load_manifest(args.manifest)
    missing = requirements(manifest, args.manifest.resolve().parent)
    print(json.dumps({"dataset_requirements_pending": missing}, ensure_ascii=False, indent=2))
    if args.strict and missing:
        raise SystemExit(2)
    if not args.validate_only:
        print(prepare(args.manifest, args.output))


if __name__ == "__main__":
    main()

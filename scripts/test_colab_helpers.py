#!/usr/bin/env python3
"""Small local tests for ZIP/layout handling and current-output export validation.
Authors: Team 8; implementation assisted by OpenAI Codex. External sources: none.
No remote files, CUDA devices or real team photographs are used.
"""
import json
import io
from email.message import Message
from pathlib import Path
import stat
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from export_colab import export_current_notebook
from resolve_dataset import copy_referenced_images, download_archive, resolve_dataset, safe_extract


class DatasetAndExportTests(unittest.TestCase):
    def setUp(self):
        self.fixture_parent = Path(__file__).resolve().parents[1] / "out" / "colab_helper_tests"
        self.fixture_parent.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=self.fixture_parent)
        self.root = Path(self.temp.name)
        self.manifest = {"schema_version": 1, "pairs": [{"left": "photos/site/a.png", "right": "photos/site/b.png",
                         "raw_left": "originals/site/a.png", "raw_right": "originals/site/b.png"}]}

    def tearDown(self):
        self.assertTrue(self.root.resolve().is_relative_to(self.fixture_parent.resolve()))
        self.temp.cleanup()

    def write_manifest(self, folder):
        folder.mkdir(parents=True, exist_ok=True)
        file = folder / "dataset_manifest.json"
        file.write_text(json.dumps(self.manifest), encoding="utf-8")
        return file

    def test_arbitrary_relative_photo_dirs_copied_with_layout(self):
        source = self.root / "source"
        manifest = self.write_manifest(source)
        for value in self.manifest["pairs"][0].values():
            path = source / value
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"fixture photo")
        workspace = self.root / "workspace"
        copied = copy_referenced_images(manifest, workspace)
        self.assertEqual(len(copied), 4)
        self.assertTrue(all((workspace / v).read_bytes() == b"fixture photo" for v in copied))

    def test_zip_wrapped_root_resolves_missing_photos_without_network(self):
        manifest = self.write_manifest(self.root / "workspace")
        archive = self.root / "dataset.zip"
        with zipfile.ZipFile(archive, "w") as output:
            for value in self.manifest["pairs"][0].values():
                output.writestr("Group08_dataset/" + value, b"fixture photo")
        with patch("resolve_dataset.download_archive", side_effect=AssertionError("must not use network")):
            result = resolve_dataset(manifest, archive_path=archive)
            self.assertEqual(result["status"], "resolved")
            self.assertFalse(result["downloaded"])
            self.assertEqual(resolve_dataset(manifest)["status"], "available")
        self.assertTrue(all((manifest.parent / v).is_file() for v in self.manifest["pairs"][0].values()))

    def test_traversal_absolute_and_symlink_archives_rejected_before_writes(self):
        for index, path in enumerate(("../escape.png", "/absolute.png", "C:/drive.png", "..\\escape.png")):
            archive = self.root / f"bad{index}.zip"
            with zipfile.ZipFile(archive, "w") as output:
                output.writestr("valid.png", b"must not be written")
                output.writestr(path, b"invalid")
            destination = self.root / f"extract{index}"
            with self.assertRaises(ValueError): safe_extract(archive, destination)
            self.assertFalse(destination.exists())
        archive = self.root / "link.zip"
        info = zipfile.ZipInfo("link")
        info.create_system = 3
        info.external_attr = (stat.S_IFLNK | 0o777) << 16
        with zipfile.ZipFile(archive, "w") as output: output.writestr(info, "../target")
        with self.assertRaises(ValueError): safe_extract(archive, self.root / "extract_link")

    def test_existing_mounted_absolute_paths_are_used_without_copy(self):
        mounted = self.root / "mounted.png"
        mounted.write_bytes(b"photo")
        self.manifest["pairs"] = [{"left": str(mounted), "right": str(mounted)}]
        manifest = self.write_manifest(self.root / "source")
        self.assertEqual(copy_referenced_images(manifest, self.root / "workspace"), [])
        self.assertEqual(resolve_dataset(manifest)["status"], "available")

    def test_direct_download_stream_and_drive_file_id_dispatch_without_network(self):
        body = io.BytesIO()
        with zipfile.ZipFile(body, "w") as archive: archive.writestr("photo.png", b"fixture")
        response = io.BytesIO(body.getvalue())
        response.headers = Message()
        response.headers["Content-Type"] = "application/zip"
        target = self.root / "direct.zip"
        with patch("resolve_dataset.urlopen", return_value=response) as opener:
            download_archive("https://example.test/dataset.zip", target)
            self.assertTrue(zipfile.is_zipfile(target))
            self.assertEqual(opener.call_count, 1)
        class FakeGdown:
            @staticmethod
            def download(**kwargs):
                self.assertEqual(kwargs["id"], "public_file_id")
                self.assertFalse(kwargs["use_cookies"])
                Path(kwargs["output"]).write_bytes(body.getvalue())
                return kwargs["output"]
        with patch("resolve_dataset.importlib.import_module", return_value=FakeGdown):
            download_archive("https://drive.google.com/file/d/public_file_id/view", self.root / "drive.zip")
        with self.assertRaises(ValueError):
            download_archive("https://drive.google.com/drive/folders/folder_id", self.root / "folder.zip")

    def test_explicit_saved_output_export_preserves_and_rejects_incomplete(self):
        notebook = {"nbformat": 4, "cells": [{"cell_type": "code", "source": "print('actual')",
                    "execution_count": 1, "outputs": [{"output_type": "stream", "text": "actual\n", "name": "stdout"}]}]}
        saved, exported = self.root / "saved.ipynb", self.root / "exports/current.ipynb"
        saved.write_text(json.dumps(notebook), encoding="utf-8")
        export_current_notebook(exported, saved)
        notebook["metadata"] = {"execution_status": "complete"}
        self.assertEqual(json.loads(exported.read_text(encoding="utf-8")), notebook)
        notebook["cells"][0]["execution_count"] = None
        saved.write_text(json.dumps(notebook), encoding="utf-8")
        with self.assertRaises(ValueError): export_current_notebook(exported, saved)
        self.assertEqual(json.loads(exported.read_text(encoding="utf-8"))["cells"][0]["execution_count"], 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)

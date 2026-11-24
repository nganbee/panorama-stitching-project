#!/usr/bin/env python3
"""Regression tests for preparation geometry, dataset validation and provenance.
Authors: Team 8; implementation assisted by OpenAI Codex. External sources: none.
These fixtures are synthetic test data, never submission photographs or results.
"""
import copy
import base64
import hashlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from contextlib import redirect_stdout

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import prepare_dataset
import benchmark
import package_submission
import provenance


class ReportingTests(unittest.TestCase):
    def setUp(self):
        self.fixture_parent = Path(__file__).resolve().parents[1] / "out" / "reporting_tests"
        self.fixture_parent.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=self.fixture_parent)
        self.root = Path(self.temp.name)
        Image.new("RGBA", (100, 50), (200, 40, 60, 255)).save(self.root / "left.png")
        Image.new("RGB", (120, 60), (20, 160, 70)).save(self.root / "right.png")
        self.manifest = {"schema_version": 1, "synthetic": True, "dataset_url": "",
                         "sites": [{"site_id": "test", "display_name": "Synthetic", "registration": "test only"}],
                         "pairs": [{"pair_id": "pair", "site_id": "test", "kind": "standard",
                                    "left": "left.png", "right": "right.png"}]}
        self.file = self.root / "manifest.json"
        self.save()

    def tearDown(self):
        self.assertTrue(self.root.resolve().is_relative_to(self.fixture_parent.resolve()))
        self.temp.cleanup()

    def save(self):
        self.file.write_text(json.dumps(self.manifest), encoding="utf-8")

    def test_same_scale_aspect_and_rgb_at_every_size(self):
        with patch.object(prepare_dataset, "RESOLUTIONS", {"small": (160, 90), "medium": (320, 180), "large": (480, 270)}):
            result = prepare_dataset.prepare(self.file, self.root / "prepared")
        prepared = json.loads(result.read_text(encoding="utf-8"))
        self.assertEqual(len(prepared["prepared_pairs"]), 3)
        for pair in prepared["prepared_pairs"]:
            for key, source in (("left", (100, 50)), ("right", (120, 60))):
                with Image.open(result.parent / pair[key]) as image:
                    self.assertEqual(image.mode, "RGB")
                    self.assertLessEqual(image.width, pair["target_box"][0])
                    self.assertLessEqual(image.height, pair["target_box"][1])
                    self.assertLessEqual(abs(image.width - source[0]*pair["scale"]), .5)
                    self.assertLessEqual(abs(image.height - source[1]*pair["scale"]), .5)
                    self.assertLessEqual(abs(image.width / image.height - 2), 2 / image.height)

    def test_traversal_duplicate_and_unknown_site_rejected(self):
        original = copy.deepcopy(self.manifest)
        for field, value in (("pair_id", "../escape"), ("site_id", "unknown"), ("kind", "invented")):
            self.manifest = copy.deepcopy(original)
            self.manifest["pairs"][0][field] = value; self.save()
            with self.assertRaises(ValueError): prepare_dataset.load_manifest(self.file)
        self.manifest = original
        self.manifest["pairs"].append(copy.deepcopy(original["pairs"][0])); self.save()
        with self.assertRaises(ValueError): prepare_dataset.load_manifest(self.file)

    def test_missing_photo_and_real_dataset_fields_not_silently_satisfied(self):
        self.assertGreaterEqual(len(prepare_dataset.requirements(self.manifest, self.root)), 3)
        self.manifest["pairs"][0]["right"] = "missing.png"; self.save()
        with self.assertRaises(FileNotFoundError): prepare_dataset.load_manifest(self.file)

    def test_amdahl_limit_and_invalid_timers(self):
        result = benchmark.profile({"end_to_end_ms": 10, "S1_ms": 4, "S2_ms": 4}, ["S1", "S2"])
        self.assertAlmostEqual(result["p"], .8)
        self.assertAlmostEqual(result["maximum_speedup"], 5)
        self.assertEqual(benchmark.profile({"end_to_end_ms": 10, "S1_ms": 11}, ["S1"])["status"], "invalid")
        self.assertEqual(benchmark.profile({"end_to_end_ms": 0, "S1_ms": 0}, ["S1"])["status"], "invalid")

    def test_unverified_binary_rejected_before_any_run(self):
        binary = self.root / "fake_binary"
        binary.write_bytes(b"test fixture, not executable")
        prepared = self.root / "prepared.json"
        prepared.write_text(json.dumps({"schema_version": 1, "prepared_pairs": []}), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "build_info"):
            benchmark.benchmark(prepared, binary, self.root / "benchmark")
        (self.root / "build_info.json").write_text(json.dumps({"source_sha256": "old", "executables": {}}), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "stale"):
            benchmark.benchmark(prepared, binary, self.root / "benchmark")

    def prepared_fixture(self):
        with patch.object(prepare_dataset, "RESOLUTIONS", {"small": (160, 90)}):
            prepared = prepare_dataset.prepare(self.file, self.root / "prepared")
        binary = self.root / "fixture_binary"
        binary.write_bytes(b"mocked test executable, never a submitted program")
        project = Path(benchmark.__file__).resolve().parents[1]
        stamp = {"source_sha256": provenance.source_sha256(project),
                 "implementation_sha256": provenance.implementation_sha256(project),
                 "executables": {"stitch": {"path": str(binary), "sha256": hashlib.sha256(binary.read_bytes()).hexdigest()}}}
        (self.root / "build_info.json").write_text(json.dumps(stamp), encoding="utf-8")
        return prepared, binary

    def mock_pipeline(self, command, **kwargs):
        left, right = prepare_dataset.image_identity(command[1]), prepare_dataset.image_identity(command[2])
        panorama = Path(command[3])
        Image.new("RGB", (32, 16), "green").save(panorama)
        metrics = {"status": "ok", "version": 0, "runs": 5, "warmup_runs": 1,
                   "imageA": dict(zip(("width", "height"), left["size"])),
                   "imageB": dict(zip(("width", "height"), right["size"])),
                   "panorama": {"width": 32, "height": 16},
                   "homography_B_to_A": [1, 0, 0, 0, 1, 0, 0, 0, 1],
                   "features": {"matches": 12, "inliers": 10},
                   "stage_timings_ms": {s: 1 for s in benchmark.STAGES},
                   "timings_ms": {"end_to_end": 10, "h2d": 0, "d2h": 0, "kernel_total": 0}}
        metrics_file = Path(command[command.index("--json")+1])
        metrics_file.write_text(json.dumps(metrics), encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, "fixture", "")

    def test_prepared_bytes_and_dimensions_rejected_before_execution(self):
        prepared, binary = self.prepared_fixture()
        data = json.loads(prepared.read_text(encoding="utf-8"))
        left = prepared.parent / data["prepared_pairs"][0]["left"]
        Image.new("RGB", (2, 2), "blue").save(left)
        with patch.object(benchmark.subprocess, "run") as execute:
            with self.assertRaisesRegex(ValueError, "Prepared image changed"):
                benchmark.benchmark(prepared, binary, self.root / "benchmark")
            execute.assert_not_called()
        data["prepared_pairs"][0]["prepared_image_sha256"]["left"] = prepare_dataset.image_identity(left)["sha256"]
        prepared.write_text(json.dumps(data), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "dimensions"):
            benchmark.verify_prepared_pair(prepared, data["prepared_pairs"][0])

    def test_cli_dimensions_and_after_run_mutation_rejected(self):
        for mode in ("cli_dimensions", "changed_after_run"):
            prepared, binary = self.prepared_fixture()
            def execute(command, **kwargs):
                result = self.mock_pipeline(command, **kwargs)
                if mode == "cli_dimensions":
                    file = Path(command[command.index("--json")+1])
                    value = json.loads(file.read_text(encoding="utf-8"))
                    value["imageA"]["width"] = 2
                    file.write_text(json.dumps(value), encoding="utf-8")
                else:
                    Image.new("RGB", (2, 2), "blue").save(command[1])
                return result
            with patch.object(benchmark.subprocess, "run", execute):
                with self.assertRaisesRegex((ValueError, RuntimeError), "dimensions|Prepared image changed"):
                    benchmark.benchmark(prepared, binary, self.root / "benchmark")
            self.assertFalse((self.root / "benchmark/benchmark.json").is_file())

    def test_benchmark_records_actual_image_metrics_and_procedure(self):
        prepared, binary = self.prepared_fixture()
        with patch.object(benchmark.subprocess, "run", self.mock_pipeline), \
                patch.object(benchmark.platform, "platform", lambda: "fixture"), \
                patch.object(benchmark.platform, "processor", lambda: "fixture"), redirect_stdout(io.StringIO()):
            result = benchmark.benchmark(prepared, binary, self.root / "benchmark")
        row = result["rows"][0]
        self.assertEqual(row["status"], "ok")
        for key in ("left", "right"):
            identity = prepare_dataset.image_identity(row["prepared_image_paths"][key])
            self.assertEqual(row["prepared_image_sha256"][key], identity["sha256"])
            self.assertEqual([row[f"{key}_width"], row[f"{key}_height"]], identity["size"])
        self.assertEqual(row["panorama_size"], [32, 16])
        self.assertEqual(row["metrics_sha256"], hashlib.sha256(Path(row["metrics_path"]).read_bytes()).hexdigest())
        self.assertEqual(result["implementation_sha256"], provenance.implementation_sha256(Path(benchmark.__file__).resolve().parents[1]))

    def strict_fixture(self):
        """Internally consistent MOCK evidence only, to exercise validator acceptance."""
        project = Path(__file__).resolve().parents[1]
        root = self.root / "submission_fixture"
        root.mkdir()
        for folder in ("src", "tools", "scripts", "docs"):
            shutil.copytree(project / folder, root / folder, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        for name in ("README.md", "CMakeLists.txt", "dataset_manifest.example.json"):
            shutil.copy2(project / name, root / name)
        Image.new("RGB", (8, 4), "red").save(root / "original.png")
        manifest = {"schema_version": 1, "dataset_url": "https://example.invalid/fixture",
                    "sites": [{"site_id": f"site{x}", "display_name": "MOCK", "registration": "MOCK"} for x in range(3)],
                    "pairs": [{"pair_id": f"pair{x}", "site_id": f"site{x%3}", "kind": "challenging" if x == 3 else "standard",
                               "left": "original.png", "right": "original.png", "raw_left": "original.png", "raw_right": "original.png"} for x in range(4)]}
        (root / "dataset_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        team = {"group": "08", "members": [{"name": "MOCK", "student_id": "MOCK"}], "source_authors_confirmed": True,
                "ai_declaration": "MOCK fixture", "dataset_quality_notes": "MOCK fixture", "failure_case_analysis": "MOCK fixture"}
        (root / "team_info.json").write_text(json.dumps(team), encoding="utf-8")
        cells = [{"cell_type": "markdown", "metadata": {}, "source": f"## 3.{x} Fixture"} for x in range(1, 8)]
        cells += [{"cell_type": "code", "source": source, "metadata": {}, "execution_count": x+1, "outputs": []}
                  for x, source in enumerate(("# fixture setup", "# fixture synthetic", "# fixture benchmark"))]
        notebook = {"cells": cells, "metadata": {"execution_status": "complete"}, "nbformat": 4, "nbformat_minor": 5}
        notebook_path = root / "Group08_Midterm.ipynb"
        notebook_path.write_text(json.dumps(notebook), encoding="utf-8")
        native, procedure = provenance.source_sha256(root), provenance.implementation_sha256(root)
        def stream(value):
            return {"output_type": "stream", "name": "stdout", "text": json.dumps(value)}
        cells[-3]["outputs"] = [stream({"source_sha256": native, "implementation_sha256": procedure,
                                         "report_inputs_sha256": {name: hashlib.sha256((root/name).read_bytes()).hexdigest()
                                                                  for name in ("team_info.json", "dataset_manifest.json")}})]
        cells[-2]["outputs"] = [stream({"all_passed": True, "tests": [{"passed": True, "mean_corner_error_px": 0,
            "max_corner_error_px": 0, "H_ground_truth": [1,0,0,0,1,0,0,0,1], "H_estimated": [1,0,0,0,1,0,0,0,1]}]})]
        binary = root / "build/stitch_fixture"
        binary.parent.mkdir()
        binary.write_bytes(b"mock fixture executable, not a submitted binary")
        binary_hash = hashlib.sha256(binary.read_bytes()).hexdigest()
        stamp = {"source_sha256": native, "implementation_sha256": procedure, "commands": [["mock-compiler"]],
                 "compiler_version": "mock compiler fixture", "executables": {"stitch": {"path": str(binary), "sha256": binary_hash}}}
        (binary.parent / "build_info.json").write_text(json.dumps(stamp), encoding="utf-8")
        prepared_manifest = root / "prepared.json"
        prepared_manifest.write_text("{}", encoding="utf-8")
        input_hash = prepare_dataset.image_identity(root / "original.png")["sha256"]
        rows = []
        for pair in manifest["pairs"]:
            for resolution in ("small", "medium", "large"):
                folder = root / "out/benchmark" / pair["pair_id"] / resolution
                folder.mkdir(parents=True)
                panorama = folder / "panorama.png"
                Image.new("RGB", (12, 4), "red").save(panorama)
                row = {"pair_id": pair["pair_id"], "site_id": pair["site_id"], "resolution": resolution,
                       "version": 0, "status": "ok", "runs": 5, "warmup_runs": 1, "end_to_end_ms": 10,
                       "prepared_image_paths": {k: str(root/"original.png") for k in ("left", "right")},
                       "prepared_image_sha256": {k: input_hash for k in ("left", "right")},
                       "left_width": 8, "left_height": 4, "right_width": 8, "right_height": 4,
                       "panorama_path": str(panorama), "panorama_size": [12,4], "panorama_sha256": prepare_dataset.image_identity(panorama)["sha256"],
                       "homography_B_to_A": [1,0,0,0,1,0,0,0,1], "hot_stages": list(benchmark.STAGES)}
                row.update({f"{s}_ms": 1 for s in benchmark.STAGES})
                row["amdahl"] = benchmark.profile(row, ["S1", "S2"])
                metrics = {"status": "ok", "version": 0, "runs": 5, "warmup_runs": 1,
                           "imageA": {"width": 8, "height": 4}, "imageB": {"width": 8, "height": 4},
                           "homography_B_to_A": row["homography_B_to_A"], "stage_timings_ms": {s:1 for s in benchmark.STAGES},
                           "timings_ms": {"end_to_end": 10}}
                metrics_path = folder / "metrics.json"
                metrics_path.write_text(json.dumps(metrics), encoding="utf-8")
                row.update(metrics_path=str(metrics_path), metrics_sha256=hashlib.sha256(metrics_path.read_bytes()).hexdigest())
                rows.append(row)
        result = {"synthetic": False, "source_sha256": native, "implementation_sha256": procedure, "build_info": stamp,
                  "executable": str(binary), "executable_sha256": binary_hash, "source_manifest": str(prepared_manifest),
                  "prepared_manifest_sha256": hashlib.sha256(prepared_manifest.read_bytes()).hexdigest(),
                  "dataset_manifest_sha256": hashlib.sha256((root/"dataset_manifest.json").read_bytes()).hexdigest(),
                  "input_image_sha256": {p["pair_id"]:{k:input_hash for k in ("left", "right")} for p in manifest["pairs"]},
                  "accelerated_stages": ["S1", "S2"], "rows": rows}
        result_path = root / "out/benchmark/benchmark.json"
        result_path.write_text(json.dumps(result), encoding="utf-8")
        png = base64.b64encode((root/"original.png").read_bytes()).decode()
        cells[-1]["outputs"] = [stream({"benchmark_sha256": hashlib.sha256(result_path.read_bytes()).hexdigest()})]
        cells[-1]["outputs"] += [{"output_type": "display_data", "metadata": {}, "data": {"image/png": png}} for _ in range(4)]
        cells[-1]["outputs"] += [{"output_type": "display_data", "metadata": {}, "data": {"text/html": "<table><tr><td>MOCK</td></tr></table>"}} for _ in range(2)]
        notebook_path.write_text(json.dumps(notebook), encoding="utf-8")
        return root, notebook, result

    def test_strict_accepts_consistent_fixture_then_rejects_missing_evidence(self):
        root, notebook, result = self.strict_fixture()
        self.assertEqual(package_submission.pending(root, "08"), [])
        notebook["cells"] = [notebook["cells"][-3]]
        (root/"Group08_Midterm.ipynb").write_text(json.dumps(notebook), encoding="utf-8")
        issues = package_submission.pending(root, "08")
        self.assertTrue(any("section 3.1" in message for message in issues))
        self.assertTrue(any("synthetic homography" in message for message in issues))
        self.assertTrue(any("panorama outputs" in message for message in issues))

    def test_strict_rejects_invalid_timings_stale_metrics_and_missing_panorama(self):
        root, notebook, result = self.strict_fixture()
        result["rows"][0]["S3_ms"] = float("nan")
        result["rows"][1]["S2_ms"] = 2
        Path(result["rows"][2]["panorama_path"]).unlink()
        (root/"out/benchmark/benchmark.json").write_text(json.dumps(result), encoding="utf-8")
        issues = package_submission.pending(root, "08")
        self.assertTrue(any("nonfinite six-stage timing" in message for message in issues))
        self.assertTrue(any("differs from saved CLI metrics" in message for message in issues))
        self.assertTrue(any("invalid measured panorama" in message for message in issues))

    def test_strict_uses_exported_outputs_and_rejects_changed_procedure(self):
        root, notebook, result = self.strict_fixture()
        exported = root / "exported.ipynb"
        exported.write_text(json.dumps(notebook), encoding="utf-8")
        stale = copy.deepcopy(notebook)
        for cell in stale["cells"]:
            if cell["cell_type"] == "code": cell["execution_count"] = None; cell["outputs"] = []
        (root/"Group08_Midterm.ipynb").write_text(json.dumps(stale), encoding="utf-8")
        self.assertEqual(package_submission.pending(root, "08", notebook_path=exported), [])
        self.assertTrue(package_submission.pending(root, "08"))
        notebook["cells"][-1]["source"] += "\n# changed after measurement"
        exported.write_text(json.dumps(notebook), encoding="utf-8")
        issues = package_submission.pending(root, "08", notebook_path=exported)
        self.assertTrue(any("source differs" in message for message in issues))
        self.assertTrue(any("report procedure" in message for message in issues))


if __name__ == "__main__":
    unittest.main(verbosity=2)

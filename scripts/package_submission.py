#!/usr/bin/env python3
"""Package source/report with best-effort checks of saved, measured evidence.
Authors: Group 08 (names/IDs pending), AI-assisted; standard-library implementation.
Checks cannot certify authorship, site registration, dataset-link access, visual quality,
or the authenticity of manually edited outputs. The team must verify those separately.
"""
import argparse
import hashlib
import json
import math
import re
import sys
import zipfile
from pathlib import Path

from benchmark import STAGES, profile
from prepare_dataset import image_identity, load_manifest, requirements
from provenance import source_sha256, implementation_sha256, notebook_source_bytes


def saved_text(notebook):
    values = [o.get("text", "") for c in notebook.get("cells", []) for o in c.get("outputs", [])]
    return "\n".join("".join(v) if isinstance(v, list) else str(v) for v in values)


def saved_json_values(text):
    """Read JSON objects printed among command prefixes and ordinary stream text."""
    decoder, offset = json.JSONDecoder(), 0
    while offset < len(text):
        start = text.find("{", offset)
        if start < 0:
            break
        try:
            value, end = decoder.raw_decode(text[start:])
            yield value
            offset = start + end
        except json.JSONDecodeError:
            offset = start + 1


def synthetic_evidence(notebook):
    for value in saved_json_values(saved_text(notebook)):
        if not isinstance(value, dict) or value.get("all_passed") is not True or not value.get("tests"):
            continue
        valid = True
        for test in value["tests"]:
            if not isinstance(test, dict) or test.get("passed") is not True:
                valid = False; break
            for name in ("mean_corner_error_px", "max_corner_error_px"):
                number = test.get(name)
                if not isinstance(number, (int, float)) or not math.isfinite(number) or number < 0:
                    valid = False
            for name in ("H_ground_truth", "H_estimated"):
                matrix = test.get(name, [])
                if len(matrix) != 9 or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in matrix):
                    valid = False
        if valid:
            return True
    return False


def pending(root, group, check_saved_outputs=True, notebook_path=None):
    root, issues = Path(root).resolve(), []
    canonical = root / f"Group{group}_Midterm.ipynb"
    notebook_path = Path(notebook_path).resolve() if notebook_path else canonical
    notebook = {"cells": []}
    if check_saved_outputs:
        if not notebook_path.is_file():
            issues.append(f"Missing saved report {notebook_path}")
        else:
            try:
                notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
                if not isinstance(notebook, dict) or not isinstance(notebook.get("cells"), list) or any(not isinstance(c, dict) for c in notebook["cells"]):
                    raise ValueError("Notebook cells must be a list of cell objects")
            except (ValueError, OSError) as error:
                issues.append(f"Invalid saved notebook: {error}")
                notebook = {"cells": []}
    required = ["README.md", "CMakeLists.txt", "scripts/build.py", "src/core_headers/common.h",
                "src/core_headers/versions.h", "src/integration/main.cpp", "src/integration/v0_pipeline.cpp",
                "tools/synthetic_pipeline_test.cpp", "src/integration/v0_selftest.cpp", "src/libraries/stb_impl.cpp",
                "src/libraries/stb_image.h", "src/libraries/stb_image_write.h", "src/core_headers/v0_stages.h",
                "src/core_headers/report_io.h", "src/core_headers/cuda_utils.cuh", "tools/run_stage12_selftest.cpp",
                "tools/stage34_selftest.cpp", "tools/stage6_selftest.cpp", "tools/synthetic_homography_test.cpp",
                "tools/debug_keypoints.cpp", "tools/timing_stage12.cpp", "scripts/prepare_dataset.py",
                "scripts/benchmark.py", "scripts/report_utils.py", "scripts/provenance.py", "scripts/resolve_dataset.py",
                "scripts/export_colab.py", "scripts/execute_notebook.py", "scripts/report_template.py"]
    required += [f"src/stage_implementations/v0_stage{suffix}.cpp" for suffix in ("12", "34", "5", "6")]
    for name in required:
        if not (root / name).is_file():
            issues.append(f"Missing required submission file {name}")
    if check_saved_outputs:
        markdown = "\n".join("".join(c.get("source", "")) for c in notebook.get("cells", []) if c.get("cell_type") == "markdown")
        for section in range(1, 8):
            if not re.search(rf"(?m)^\s*#{{1,6}}\s+3\.{section}\b", markdown):
                issues.append(f"Report is missing required section 3.{section}")
        code = [c for c in notebook.get("cells", []) if c.get("cell_type") == "code"]
        if not code:
            issues.append("Report has no executable build/result cells")
        for index, cell in enumerate(notebook.get("cells", [])):
            if cell.get("cell_type") == "code":
                if cell.get("execution_count") is None:
                    issues.append(f"Report code cell {index} has not been executed")
                if any(o.get("output_type") == "error" for o in cell.get("outputs", [])):
                    issues.append(f"Report code cell {index} contains an execution error")
        if notebook.get("metadata", {}).get("execution_status") == "running":
            issues.append("Report execution was interrupted/incomplete")
        if not synthetic_evidence(notebook):
            issues.append("Report lacks successful synthetic homography correctness with ground truth and measured corner errors")
        if canonical.is_file() and notebook_path.is_file():
            try:
                if notebook_source_bytes(canonical) != notebook_source_bytes(notebook_path):
                    issues.append("Saved/exported notebook source differs from measured report source; synchronize before rebuilding and rerun")
            except (ValueError, OSError) as error:
                issues.append(f"Cannot fingerprint report source: {error}")
    team_path = root / "team_info.json"
    if not team_path.is_file():
        issues.append("Missing team_info.json with verified names, student IDs, authors and AI declaration")
    else:
        try:
            team = json.loads(team_path.read_text(encoding="utf-8"))
            members = team.get("members", [])
            if not members or any(not m.get("name") or not m.get("student_id") for m in members):
                issues.append("Team members have missing full names/student IDs")
            if not team.get("ai_declaration") or not team.get("source_authors_confirmed"):
                issues.append("Team must confirm source author headers and AI declaration")
            if str(team.get("group")) != group:
                issues.append("team_info group does not match submission group")
            ids = [m.get("student_id") for m in members]
            if len(ids) != len(set(ids)):
                issues.append("Student IDs must be unique")
            for name in ("dataset_quality_notes", "failure_case_analysis"):
                if not team.get(name):
                    issues.append(f"Team must fill {name} using actual results")
        except (ValueError, TypeError, AttributeError) as error:
            issues.append(f"Invalid team_info.json: {error}")
    manifest_path, manifest = root / "dataset_manifest.json", None
    if not manifest_path.is_file():
        issues.append("Missing dataset_manifest.json with actual team photographs")
    else:
        try:
            manifest = load_manifest(manifest_path)
            issues.extend(requirements(manifest, manifest_path.parent))
            if manifest.get("synthetic"):
                issues.append("Synthetic images do not fulfil registered-site dataset requirement")
        except (ValueError, KeyError, TypeError, OSError) as error:
            issues.append(f"Dataset invalid: {error}")
    native_hash = source_sha256(root)
    try:
        procedure_hash = implementation_sha256(root, notebook_path if notebook_path.is_file() else None)
    except (ValueError, OSError) as error:
        issues.append(f"Cannot fingerprint report procedure: {error}")
        procedure_hash = None
    output_text = saved_text(notebook)
    if check_saved_outputs:
        for key, value in (("source_sha256", native_hash), ("implementation_sha256", procedure_hash)):
            if not any(isinstance(v, dict) and v.get(key) == value for v in saved_json_values(output_text)):
                issues.append(f"Saved report has no current {key}; rebuild and rerun notebook")
        for name in ("team_info.json", "dataset_manifest.json"):
            file = root / name
            if file.is_file() and hashlib.sha256(file.read_bytes()).hexdigest() not in output_text:
                issues.append(f"Saved report inputs differ from current {name}; rerun notebook")
    results_path = root / "out/benchmark/benchmark.json"
    if not results_path.is_file():
        issues.append("No real dataset benchmark: rerun report with dataset_manifest.json")
        return issues
    try:
        result = json.loads(results_path.read_text(encoding="utf-8"))
        if result.get("synthetic"):
            issues.append("Real dataset timing and panoramas still pending")
        if result.get("source_sha256") != native_hash or result.get("implementation_sha256") != procedure_hash:
            issues.append("Dataset benchmark was not measured from current native sources/report procedure; rebuild and rerun")
        stamp, executable = result.get("build_info", {}), Path(result.get("executable", ""))
        binary_entry = next((e for e in stamp.get("executables", {}).values() if Path(e.get("path", "")).resolve() == executable.resolve()), None)
        valid_build = (stamp.get("source_sha256") == native_hash and stamp.get("implementation_sha256") == procedure_hash
                       and bool(stamp.get("commands")) and bool(stamp.get("compiler_version")) and executable.is_file() and binary_entry
                       and binary_entry.get("sha256") == result.get("executable_sha256")
                       and hashlib.sha256(executable.read_bytes()).hexdigest() == result.get("executable_sha256"))
        current_stamp = executable.parent / "build_info.json"
        if not valid_build or not current_stamp.is_file() or json.loads(current_stamp.read_text(encoding="utf-8")) != stamp:
            issues.append("Dataset benchmark has unverifiable/stale build provenance; rebuild and measure")
        if manifest_path.is_file() and result.get("dataset_manifest_sha256") != hashlib.sha256(manifest_path.read_bytes()).hexdigest():
            issues.append("Benchmark does not match current dataset manifest")
        prepared_manifest = Path(result.get("source_manifest", ""))
        if not prepared_manifest.is_file() or hashlib.sha256(prepared_manifest.read_bytes()).hexdigest() != result.get("prepared_manifest_sha256"):
            issues.append("Prepared manifest missing/changed since measurement")
        rows = result.get("rows", [])
        successful = [r for r in rows if r.get("status") == "ok" and r.get("version") == 0]
        for resolution in ("small", "medium", "large"):
            if not any(r.get("resolution") == resolution for r in successful):
                issues.append(f"Missing measured V0 timing for {resolution}")
        for row in successful:
            label = f"{row.get('pair_id')}/{row.get('resolution')}"
            times = [row.get(f"{s}_ms") for s in STAGES] + [row.get("end_to_end_ms")]
            valid_times = all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) and v >= 0 for v in times)
            if not valid_times or row.get("end_to_end_ms", 0) <= 0:
                issues.append(f"Missing/nonfinite six-stage timing: {label}")
            elif any(s not in STAGES for s in result.get("accelerated_stages", [])) or not result.get("accelerated_stages"):
                issues.append("Missing/invalid accelerated-stage plan")
            else:
                expected, amdahl = profile(row, result["accelerated_stages"]), row.get("amdahl", {})
                if expected.get("status") != "ok" or any(amdahl.get(k) != expected.get(k) for k in expected):
                    issues.append(f"Missing/inconsistent Amdahl evidence: {label}")
                if row.get("hot_stages") != sorted(STAGES, key=lambda s: row[f"{s}_ms"], reverse=True):
                    issues.append(f"Missing/inconsistent hotspot ranking: {label}")
            if not isinstance(row.get("runs"), int) or row["runs"] < 5 or not isinstance(row.get("warmup_runs"), int) or row["warmup_runs"] < 1:
                issues.append(f"Missing measured-run/warm-up counts: {label}")
            for key in ("left", "right"):
                try:
                    identity = image_identity(row.get("prepared_image_paths", {})[key])
                    if identity["sha256"] != row.get("prepared_image_sha256", {}).get(key) or identity["size"] != [row.get(f"{key}_width"), row.get(f"{key}_height")]:
                        issues.append(f"Prepared measurement input changed: {label}/{key}")
                except (OSError, KeyError, TypeError, ValueError):
                    issues.append(f"Missing/invalid prepared measurement input: {label}/{key}")
            try:
                panorama = image_identity(row["panorama_path"])
                if panorama["sha256"] != row.get("panorama_sha256") or panorama["size"] != row.get("panorama_size"):
                    issues.append(f"Panorama changed since measurement: {label}")
            except (OSError, KeyError, TypeError, ValueError):
                issues.append(f"Missing/invalid measured panorama: {label}")
            metrics = Path(row.get("metrics_path", ""))
            if not metrics.is_file() or hashlib.sha256(metrics.read_bytes()).hexdigest() != row.get("metrics_sha256"):
                issues.append(f"Missing/changed CLI metrics: {label}")
            else:
                measured = json.loads(metrics.read_text(encoding="utf-8"))
                consistent = (measured.get("status") == "ok" and measured.get("version") == 0
                              and measured.get("runs") == row.get("runs") and measured.get("warmup_runs") == row.get("warmup_runs")
                              and measured.get("homography_B_to_A") == row.get("homography_B_to_A")
                              and all(measured.get("stage_timings_ms", {}).get(s) == row.get(f"{s}_ms") for s in STAGES)
                              and measured.get("timings_ms", {}).get("end_to_end") == row.get("end_to_end_ms"))
                for side, name in (("left", "imageA"), ("right", "imageB")):
                    consistent = consistent and measured.get(name, {}).get("width") == row.get(f"{side}_width") and measured.get(name, {}).get("height") == row.get(f"{side}_height")
                if not consistent:
                    issues.append(f"Benchmark row differs from saved CLI metrics: {label}")
                matrix = measured.get("homography_B_to_A", [])
                if len(matrix) != 9 or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in matrix):
                    issues.append(f"Missing/nonfinite measured homography: {label}")
        if manifest:
            for site in manifest.get("sites", []):
                if not any(r.get("site_id") == site["site_id"] for r in successful):
                    issues.append(f"Missing panorama for {site['site_id']}")
            for pair in manifest.get("pairs", []):
                for key in ("left", "right"):
                    recorded = result.get("input_image_sha256", {}).get(pair["pair_id"], {}).get(key)
                    if image_identity(manifest_path.parent / pair[key])["sha256"] != recorded:
                        issues.append(f"Benchmark uses a different original input {pair['pair_id']}/{key}")
                for resolution in ("small", "medium", "large"):
                    candidates = [r for r in rows if r.get("pair_id") == pair["pair_id"] and r.get("site_id") == pair["site_id"] and r.get("resolution") == resolution and r.get("version") == 0]
                    if not candidates:
                        issues.append(f"Missing measured pair {pair['pair_id']}/{resolution}")
                    elif pair.get("kind") == "standard" and not any(r.get("status") == "ok" for r in candidates):
                        issues.append(f"Standard pair failed {pair['pair_id']}/{resolution}")
                    elif any(r.get("status") != "ok" and not r.get("error") for r in candidates):
                        issues.append(f"Failed challenging pair lacks failure evidence {pair['pair_id']}/{resolution}")
        if check_saved_outputs:
            results_hash = hashlib.sha256(results_path.read_bytes()).hexdigest()
            evidence_cells = [c for c in notebook.get("cells", []) if any(isinstance(v, dict) and v.get("benchmark_sha256") == results_hash for v in saved_json_values(saved_text({"cells": [c]})))]
            image_count = sum("image/png" in o.get("data", {}) for c in evidence_cells for o in c.get("outputs", []))
            table_count = sum("text/html" in o.get("data", {}) for c in evidence_cells for o in c.get("outputs", []))
            sites_count = len(manifest.get("sites", [])) if manifest else 3
            if not evidence_cells or image_count < sites_count + 1 or table_count < 2:
                issues.append("Saved real-dataset benchmark lacks matching timing/analysis tables, plot and per-site panorama outputs; rerun and save")
    except (ValueError, KeyError, TypeError, OSError) as error:
        issues.append(f"Invalid benchmark evidence: {error}")
    return issues


def package(root, group, destination, strict=False, notebook_path=None):
    root, destination = Path(root).resolve(), Path(destination).resolve()
    notebook = Path(notebook_path).resolve() if notebook_path else root / f"Group{group}_Midterm.ipynb"
    issues = pending(root, group, notebook_path=notebook)
    print(json.dumps({"submission_pending": issues, "strict": strict}, ensure_ascii=False, indent=2))
    if strict and issues:
        raise SystemExit("Strict submission check failed; complete listed evidence and rerun/save notebook.")
    files = [root / "README.md", root / "CMakeLists.txt", root / "dataset_manifest.example.json"]
    files += [root / name for name in ("dataset_manifest.json", "team_info.json") if (root / name).is_file()]
    for directory in ("src", "tools", "scripts", "docs"):
        files.extend(p for p in (root / directory).rglob("*") if p.is_file() and "__pycache__" not in p.parts and p.suffix not in (".pyc", ".o", ".obj", ".exe"))
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        if notebook.is_file():
            archive.write(notebook, f"Group{group}_Midterm.ipynb")
        for file in sorted(set(files)):
            if file.is_file():
                archive.write(file, file.relative_to(root).as_posix())
    print(f"Wrote {destination}. This archive is {'ready after the team verifies authorship/site/link/visual quality' if not issues else 'a source/report draft; pending items listed above'}.")
    return destination


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--group", default="08")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--notebook", type=Path, help="Actual saved/exported notebook with current Colab outputs")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    if not re.fullmatch(r"[0-9]{2,3}", args.group):
        parser.error("group must contain two or three digits")
    package(args.root, args.group, args.output or args.root / f"Group{args.group}_Midterm.zip", args.strict, args.notebook)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Reproducible six-stage benchmark. Authors: Group 08 (names/IDs pending), AI-assisted.
External sources: Python standard-library APIs; no external implementation copied.
"""
import argparse
import csv
import hashlib
import json
import math
import platform
import subprocess
import sys
import time
from pathlib import Path

from prepare_dataset import image_identity
from provenance import source_sha256, implementation_sha256

STAGES = ("S1", "S2", "S3", "S4", "S5", "S6")


def verify_prepared_pair(manifest_path, pair):
    identities = {}
    for key in ("left", "right"):
        identity = image_identity(manifest_path.parent / pair[key])
        if identity["sha256"] != pair.get("prepared_image_sha256", {}).get(key):
            raise ValueError(f"Prepared image changed or has no fingerprint: {pair['pair_id']}/{pair['resolution']}/{key}; prepare again")
        if identity["size"] != pair.get(f"{key}_size"):
            raise ValueError(f"Prepared dimensions do not match decoded image: {pair['pair_id']}/{key}")
        identities[key] = identity
    return identities


def profile(row, accelerate):
    total = row["end_to_end_ms"]
    chosen = sum(row[f"{s}_ms"] for s in accelerate)
    if total <= 0:
        return {"status": "invalid", "reason": "Nonpositive end-to-end time"}
    p = chosen / total
    if p > 1.000001:
        return {"status": "invalid", "reason": "Selected stage timers exceed end-to-end timer", "p": p}
    p = min(p, 1.0)
    limit = None if p == 1 else 1 / (1 - p)
    return {"status": "ok", "stages": list(accelerate), "p": p,
            "maximum_speedup": limit, "limit_is_infinite": p == 1}


def benchmark(manifest_path, executable, output, runs=5, versions=(0,), accelerate=("S1", "S2", "S3", "S4", "S6")):
    manifest_path = Path(manifest_path).resolve()
    executable = Path(executable).resolve()
    output = Path(output).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    root = Path(__file__).resolve().parents[1]
    source_hash = source_sha256(root)
    procedure_hash = implementation_sha256(root)
    stamp_path = executable.parent / "build_info.json"
    if not stamp_path.is_file():
        raise ValueError("Missing build_info.json; build with scripts/build.py before measuring")
    stamp = json.loads(stamp_path.read_text(encoding="utf-8"))
    binary_hash = hashlib.sha256(executable.read_bytes()).hexdigest()
    binary_stamp = next((entry for entry in stamp.get("executables", {}).values()
                         if Path(entry["path"]).resolve() == executable), None)
    if (stamp.get("source_sha256") != source_hash or stamp.get("implementation_sha256") != procedure_hash
            or not binary_stamp or binary_stamp.get("sha256") != binary_hash):
        raise ValueError("Build provenance is stale or invalid; rebuild with scripts/build.py")
    if runs < 5 or not versions or any(v not in (0, 1, 2, 3) for v in versions):
        raise ValueError("At least five runs and valid version numbers (0..3) are required")
    if not accelerate or any(s not in STAGES for s in accelerate) or len(set(accelerate)) != len(accelerate):
        raise ValueError("Accelerated stages must be a nonempty unique subset of S1..S6")
    if any(v != 0 for v in versions) and not stamp.get("cuda_enabled"):
        raise ValueError("GPU benchmark requires a verified CUDA build")
    if manifest.get("schema_version") != 1:
        raise ValueError("Invalid prepared manifest schema")
    prepared_manifest_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for pair in manifest["prepared_pairs"]:
        pid = pair["pair_id"]
        if not pid or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in pid):
            raise ValueError("Unsafe prepared pair_id")
        if pair["resolution"] not in ("small", "medium", "large"):
            raise ValueError("Unknown prepared resolution")
        identities = verify_prepared_pair(manifest_path, pair)
        for version in versions:
            verify_prepared_pair(manifest_path, pair)
            pair_dir = (output / pid / pair["resolution"] / f"v{version}").resolve()
            if not pair_dir.is_relative_to(output):
                raise ValueError("Output path escapes benchmark directory")
            pair_dir.mkdir(parents=True, exist_ok=True)
            metrics_file = pair_dir / "metrics.json"
            panorama = pair_dir / "panorama.png"
            # A previous run cannot supply this run's metrics or panorama.
            for stale in (metrics_file, panorama):
                if stale.is_file():
                    stale.unlink()
            command = [str(executable), str(manifest_path.parent / pair["left"]),
                       str(manifest_path.parent / pair["right"]), str(panorama),
                       "--version", str(version), "--runs", str(runs), "--json", str(metrics_file)]
            completed = subprocess.run(command, capture_output=True, text=True)
            (pair_dir / "stdout.txt").write_text(completed.stdout, encoding="utf-8")
            (pair_dir / "stderr.txt").write_text(completed.stderr, encoding="utf-8")
            row = {"pair_id": pair["pair_id"], "site_id": pair["site_id"], "kind": pair["kind"],
                   "resolution": pair["resolution"], "version": version,
                   "synthetic": bool(manifest.get("synthetic", False)),
                   "left_width": identities["left"]["size"][0], "left_height": identities["left"]["size"][1],
                   "right_width": identities["right"]["size"][0], "right_height": identities["right"]["size"][1],
                   "scale": pair["scale"], "returncode": completed.returncode,
                   "metrics_path": str(metrics_file), "panorama_path": str(panorama),
                   "input_image_sha256": pair.get("input_image_sha256", {}),
                   "prepared_image_sha256": {key: identities[key]["sha256"] for key in ("left", "right")},
                   "prepared_image_paths": {key: str((manifest_path.parent / pair[key]).resolve()) for key in ("left", "right")},
                   "status": "failed"}
            metrics = json.loads(metrics_file.read_text(encoding="utf-8")) if metrics_file.is_file() else {}
            if completed.returncode == 0 and metrics.get("status") == "ok" and panorama.is_file():
                if metrics.get("version") != version:
                    raise RuntimeError("CLI returned a different pipeline version")
                for key, name in (("left", "imageA"), ("right", "imageB")):
                    measured = metrics.get(name, {})
                    if [measured.get("width"), measured.get("height")] != identities[key]["size"]:
                        raise RuntimeError(f"CLI dimensions differ from the prepared input: {pair['pair_id']}/{key}")
                if metrics.get("runs") != runs or metrics.get("warmup_runs", 0) < 1:
                    raise RuntimeError("CLI run/warm-up counts do not match the benchmark")
                row.update({f"{stage}_ms": float(metrics["stage_timings_ms"][stage]) for stage in STAGES})
                for name in ("end_to_end", "h2d", "d2h", "kernel_total"):
                    row[f"{name}_ms"] = float(metrics["timings_ms"][name])
                time_keys = [f"{s}_ms" for s in STAGES] + [f"{name}_ms" for name in ("end_to_end", "h2d", "d2h", "kernel_total")]
                if not all(math.isfinite(row[k]) and row[k] >= 0 for k in time_keys) or row["end_to_end_ms"] <= 0:
                    raise RuntimeError("Invalid stage timing in CLI output")
                row.update(metrics["features"])
                row["homography_B_to_A"] = metrics["homography_B_to_A"]
                if len(row["homography_B_to_A"]) != 9 or not all(math.isfinite(float(v)) for v in row["homography_B_to_A"]):
                    raise RuntimeError("CLI returned a nonfinite/invalid homography")
                panorama_identity = image_identity(panorama)
                row["panorama_size"] = panorama_identity["size"]
                row["panorama_sha256"] = panorama_identity["sha256"]
                reported_panorama = metrics.get("panorama", {})
                if [reported_panorama.get("width"), reported_panorama.get("height")] != panorama_identity["size"]:
                    raise RuntimeError("CLI panorama dimensions differ from the saved image")
                row.update({"status": "ok", "runs": metrics["runs"], "warmup_runs": metrics["warmup_runs"],
                            "hot_stages": sorted(STAGES, key=lambda s: row[f"{s}_ms"], reverse=True)})
                row["amdahl"] = profile(row, accelerate)
            else:
                row["error"] = metrics.get("error", completed.stderr.strip() or "Missing metrics output")
            verify_prepared_pair(manifest_path, pair)
            if metrics_file.is_file():
                row["metrics_sha256"] = hashlib.sha256(metrics_file.read_bytes()).hexdigest()
            rows.append(row)
            print(json.dumps(row, ensure_ascii=False))
    # Comparing versions uses the same prepared pair, resolution and measured session.
    baselines = {(r["pair_id"], r["resolution"]): r for r in rows if r["version"] == 0 and r["status"] == "ok"}
    for row in rows:
        base = baselines.get((row["pair_id"], row["resolution"]))
        if row["status"] == "ok" and base and row["end_to_end_ms"] > 0:
            row["speedup_vs_v0"] = base["end_to_end_ms"] / row["end_to_end_ms"]
    if (source_sha256(root) != source_hash or implementation_sha256(root) != procedure_hash
            or hashlib.sha256(executable.read_bytes()).hexdigest() != binary_hash):
        raise RuntimeError("Source or executable changed during measurement; rebuild and rerun")
    if hashlib.sha256(manifest_path.read_bytes()).hexdigest() != prepared_manifest_hash:
        raise RuntimeError("Prepared manifest changed during measurement; prepare and rerun")
    for pair in manifest["prepared_pairs"]:
        verify_prepared_pair(manifest_path, pair)
    result = {"schema_version": 1, "created_unix": time.time(), "platform": platform.platform(),
              "processor": platform.processor(), "executable": str(executable),
              "source_manifest": str(manifest_path), "dataset_url": manifest.get("dataset_url", ""),
              "prepared_manifest_sha256": prepared_manifest_hash,
              "dataset_manifest_sha256": manifest.get("input_manifest_sha256"),
              "source_sha256": source_hash, "implementation_sha256": procedure_hash,
              "executable_sha256": binary_hash, "build_info": stamp,
              "input_image_sha256": {pair["pair_id"]:pair.get("input_image_sha256", {}) for pair in manifest["prepared_pairs"]},
              "synthetic": bool(manifest.get("synthetic", False)), "accelerated_stages": list(accelerate),
              "timing_convention": "One CLI warm-up, arithmetic mean of measured pipeline runs; I/O excluded; S1-S3 include both images.",
              "rows": rows}
    (output / "benchmark.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    columns = ["pair_id", "site_id", "kind", "resolution", "version", "synthetic", "status",
               "left_width", "left_height", "right_width", "right_height", "runs", "warmup_runs"]
    columns += [f"{s}_ms" for s in STAGES] + ["end_to_end_ms", "h2d_ms", "d2h_ms", "kernel_total_ms",
                "matches", "inliers", "inlier_ratio", "speedup_vs_v0", "error"]
    with (output / "benchmark.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return result


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--executable", type=Path, default=Path("build/stitch"))
    parser.add_argument("--output", type=Path, default=Path("out/benchmark"))
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--versions", type=int, nargs="+", default=[0])
    parser.add_argument("--accelerate", choices=STAGES, nargs="+", default=["S1", "S2", "S3", "S4", "S6"])
    args = parser.parse_args()
    if args.runs < 5:
        parser.error("Use at least five timed runs")
    result = benchmark(args.manifest, args.executable, args.output, args.runs, args.versions, args.accelerate)
    if any(row["status"] != "ok" for row in result["rows"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()

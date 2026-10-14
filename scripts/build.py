#!/usr/bin/env python3
# Authors: Team 8; implementation assisted by OpenAI Codex.
# External sources: none. Uses the installed compiler, never notebook copies.
"""Build the shared source tree on Colab/Linux or Windows; CUDA is opt-in."""
import argparse
import json
import hashlib
import os
from pathlib import Path
import shutil
import subprocess

from provenance import implementation_sha256, source_sha256

ROOT = Path(__file__).resolve().parents[1]
CPU_SOURCES = ["src/integration/v0_pipeline.cpp", "src/stage_implementations/v0_stage12.cpp",
               "src/stage_implementations/v0_stage34.cpp", "src/stage_implementations/v0_stage5.cpp",
               "src/stage_implementations/v0_stage6.cpp", "src/libraries/stb_impl.cpp"]
TARGETS = {"stitch": "src/integration/main.cpp", "stage12_selftest": "tools/run_stage12_selftest.cpp",
           "stage34_selftest": "tools/stage34_selftest.cpp", "stage6_selftest": "tools/stage6_selftest.cpp",
           "synthetic_homography_test": "tools/synthetic_homography_test.cpp",
           "synthetic_pipeline_test": "tools/synthetic_pipeline_test.cpp",
           "debug_keypoints": "tools/debug_keypoints.cpp", "timing_stage12": "tools/timing_stage12.cpp",
           "cuda_host_math_selftest": "src/cuda/host_math_selftest.cpp"}

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--compiler", default=os.environ.get("CXX", "g++"))
    ap.add_argument("--cuda", action="store_true", help="Use nvcc; requires CUDA toolkit and host compiler")
    ap.add_argument("--build-dir", type=Path, default=ROOT / "build")
    ap.add_argument("--target", choices=sorted(TARGETS), action="append", help="Default: all targets")
    args = ap.parse_args()
    compiler = shutil.which("nvcc" if args.cuda else args.compiler)
    if not compiler:
        ap.error("Compiler not found: " + ("nvcc" if args.cuda else args.compiler))
    build = args.build_dir.resolve()
    build.mkdir(parents=True, exist_ok=True)
    source_hash = source_sha256(ROOT)
    implementation_hash = implementation_sha256(ROOT)
    sources = list(CPU_SOURCES) + ["src/integration/v0_selftest.cpp"]
    if args.cuda:
        sources.append("src/cuda/cuda_pipeline.cu")
    base = [compiler, "-O2", "-std=c++17", "-Isrc/core_headers", "-Isrc/libraries"]
    if args.cuda:
        base += ["-DPANORAMA_WITH_CUDA=1", "--fmad=false"]
    else:
        base += ["-Wall", "-Wextra"]
    objects = []
    commands = []
    for source in sources:
        obj = build / (Path(source).stem + ".o")
        command = base + ["-c", source, "-o", str(obj)]
        commands.append(command)
        print("Compile", source, flush=True)
        subprocess.run(command, cwd=ROOT, check=True)
        objects.append(str(obj))
    suffix = ".exe" if os.name == "nt" else ""
    executables = {}
    for target in args.target or list(TARGETS):
        command = base + [TARGETS[target], *objects, "-o", str(build / (target + suffix))]
        commands.append(command)
        print("Link", target, flush=True)
        subprocess.run(command, cwd=ROOT, check=True)
        binary = build / (target + suffix)
        executables[target] = {"path": str(binary), "sha256": hashlib.sha256(binary.read_bytes()).hexdigest()}
    version = subprocess.check_output([compiler, "--version"], text=True, errors="replace")
    if (source_sha256(ROOT) != source_hash or
            implementation_sha256(ROOT) != implementation_hash):
        raise RuntimeError("Source/report procedure changed during compilation. "
                           "Run build.py again before measuring.")
    (build / "build_info.json").write_text(json.dumps({"schema_version": 1, "compiler": compiler,
        "compiler_version": version, "cuda_enabled": args.cuda, "commands": commands,
        "source_sha256": source_hash, "implementation_sha256": implementation_hash,
        "executables": executables}, indent=2), encoding="utf-8")
    print("Built sources successfully in", build)

if __name__ == "__main__":
    main()

"""V0/V3/V4 reports and correctness on the 12 project cases (Colab/Linux).

Run from the repository root after building the CLI and parity executable.
Every CLI invocation performs one warm-up and averages >=5 measured runs.
Failures remain failures in the summary; no failed parity is reported as PASS.
"""
import argparse
import csv
import datetime
import json
from pathlib import Path
import subprocess
import sys


def execute(args, log):
    result = subprocess.run([str(a) for a in args], text=True, capture_output=True)
    log.write_text(result.stdout + result.stderr, encoding="utf-8")
    print(result.stdout, end="", flush=True)
    if result.stderr:
        print(result.stderr, file=sys.stderr, end="", flush=True)
    return result.returncode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", default="build/stitch_v4")
    parser.add_argument("--parity", default="build/test_v4_pipeline")
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--output", default=None)
    parser.add_argument("--reverse-order", action="store_true",
                        help="Second sweep: measure V4 before V3 to assess run-order bias")
    args = parser.parse_args()
    if args.runs < 5:
        parser.error("--runs must be >=5")
    binary, parity = Path(args.binary).resolve(), Path(args.parity).resolve()
    if not binary.is_file() or not parity.is_file():
        parser.error("Build stitch_v4 and test_v4_pipeline first")
    root = Path(args.output or ("out/v4_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S")))
    root.mkdir(parents=True, exist_ok=False)
    metadata = {"runs": args.runs, "warmup_runs": 1, "binary": str(binary),
                "baseline": "merged V3 (V2 shared-memory S4; V3 S6)", "argv": sys.argv}
    for key, command in {"commit": ["git", "rev-parse", "HEAD"],
                         "worktree": ["git", "status", "--short"],
                         "gpu": ["nvidia-smi"], "nvcc": ["nvcc", "--version"]}.items():
        try:
            result = subprocess.run(command, text=True, capture_output=True)
            metadata[key] = result.stdout + result.stderr
        except OSError as error:
            metadata[key] = str(error)
    (root / "environment.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print("Reports:", root.resolve(), flush=True)
    rows = []
    for location in ("location1", "location2", "location3", "challenge"):
        for label, folder in (("720p", "small_720p"), ("1080p", "medium_1080p"), ("4k", "large_4k")):
            case = f"{location}_{label}"
            left, right = (Path("dataset") / location / folder / f"{i}.jpg" for i in (1, 2))
            row = {"case": case, "status": "failed", "parity": "not_run"}
            print("\nCASE", case, flush=True)
            if not left.is_file() or not right.is_file():
                row["error"] = "Missing dataset pair"
                rows.append(row)
                continue
            ok = execute([parity, left, right], root / f"{case}_parity.log") == 0
            row["parity"] = "pass" if ok else "failed"
            reports = {}
            # Use --reverse-order on a second sweep to assess thermal/order bias.
            for version in ((0, 4, 3) if args.reverse_order else (0, 3, 4)):
                report_path = root / f"{case}_v{version}.json"
                code = execute([binary, left, right, root / f"{case}_v{version}.png",
                                "--version", version, "--runs", args.runs, "--json", report_path],
                               root / f"{case}_v{version}.log")
                if code != 0 or not report_path.is_file():
                    ok = False
                    continue
                report = json.loads(report_path.read_text(encoding="utf-8"))
                if (report.get("status") != "ok" or report.get("runs", 0) < 5 or
                        report.get("warmup_runs", 0) < 1):
                    ok = False
                    continue
                reports[version] = report
                for key in ("blur", "gray", "h2d", "d2h", "kernel_total", "end_to_end"):
                    row[f"v{version}_{key}_ms"] = report["timings_ms"][key]
                for stage, ms in report["stage_timings_ms"].items():
                    row[f"v{version}_{stage}_ms"] = ms
            if len(reports) == 3:
                t3, t4 = reports[3]["timings_ms"], reports[4]["timings_ms"]
                row["blur_speedup_v3_v4"] = t3["blur"] / t4["blur"] if t4["blur"] > 0 else None
                row["e2e_speedup_v3_v4"] = t3["end_to_end"] / t4["end_to_end"] if t4["end_to_end"] > 0 else None
                row["e2e_speedup_v0_v4"] = reports[0]["timings_ms"]["end_to_end"] / t4["end_to_end"]
                row["v3_blur_share_e2e_pct"] = 100 * t3["blur"] / t3["end_to_end"]
                row["v3_blur_share_kernel_pct"] = 100 * t3["blur"] / t3["kernel_total"]
                for version in (3, 4):
                    hottest = max(reports[version]["stage_timings_ms"], key=reports[version]["stage_timings_ms"].get)
                    row[f"v{version}_bottleneck_stage"] = hottest
                row["status"] = "ok" if ok else "failed"
            else:
                row["error"] = "One or more CLI benchmarks failed; see case logs"
            rows.append(row)
            # Persist progress even if the Colab session disconnects later.
            (root / "summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    (root / "summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    with (root / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(key for row in rows for key in row)))
        writer.writeheader()
        writer.writerows(rows)
    print("\nSummary:", root / "summary.csv")
    return 0 if len(rows) == 12 and all(row["status"] == "ok" for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())

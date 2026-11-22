# V2 — S4 shared-memory descriptor tiling

Owner: Khánh. Source, verification and notebook preparation assisted by OpenAI
Codex. Algorithms and acceptance criteria derive from the team's V0/V1 and
`docs/PLAN.md`. No GPU execution or speedup is claimed by this document.

## Scope

V2 changes only S4 candidate access to shared-memory tiling. S1–S3, S5–S6,
algorithms, parameters, matching policy and host/device data flow inherit V1.
V1 implementation files and notebook are unchanged. The common registry exposes
V2 only when `PANORAMA_WITH_CUDA` and `PANORAMA_WITH_V2` are enabled.

Each block has 128 threads, one query per thread. It cooperatively loads 128
candidate descriptors into 4096 bytes of shared memory, scans them in ascending
index order, then synchronizes before overwriting the tile. Inactive query
threads still participate in barriers. Both matching directions use tiling;
ratio test, cross-check, tie-break and ordered compaction remain unchanged.
The candidate count bounds the final partial tile. Cache reuse in V1 and barrier
overhead may make V2 slower; actual baseline timing and profiler evidence decide.

## Notebook workflow

`test_v2.ipynb` retains the V1 style of executable cells, Markdown, panorama,
tables and plots, with the content focused on the optimization:

1. Setup source/GPU/dataset; validate all 12 pairs and required source files.
2. Build V0/V1/V2 plus stage, integration, benchmark and verification binaries.
3. Measure V0/V1 in one session, then collect actual Nsight evidence for V1.
4. Explain the idea, inherited stages, kernel mapping and matching invariants.
5. Run S4 parity, memcheck and synccheck. Other standalone V1 tests are optional.
6. Check synthetic integration and one real pair before full-dataset evaluation.
7. Verify and benchmark V2 across all 12 cases using the same session and inputs.
8. Report correctness, raw sample dispersion, speedup, breakdown and gallery.
9. Analyze faster/slower/failed cases with profiling evidence.
10. List outstanding acceptance checks, deliverables, references and AI usage.

Every optimized version requires the five report parts: bottleneck, idea,
implementation, result, analysis. Timing alone does not replace profiler output.
Profiler and sanitizer results are excluded from speedup calculations. Missing
tools or permission errors remain NOT VERIFIED with actual logs.

## Source and tools

All V2 support tools are grouped in `tools/v2/`. Production implementations
remain in `src/`; original V0/V1 tools remain at their existing paths.

| File | Role |
|---|---|
| `src/stage_implementations/v2_stage4.cu` | Independent tiled matcher |
| `src/core_headers/v2_stages.h` | Matcher interface |
| `src/cuda/v2_pipeline.cu`, `.h` | V2 pipeline entry point |
| `tools/v2/build.sh` | Common build flags and source lists |
| `tools/v2/test_matching.cpp` | Exact V0/V1/V2 stage parity, tile/ratio/tie/cross-check cases |
| `tools/v2/test_pipeline.cpp` | Synthetic integration, direct V1/V2 and failure contracts |
| `tools/v2/benchmark_pair.cpp` | Production pipelines, warm-up plus timed samples, average metrics and panorama |
| `tools/v2/benchmark.py` | Two-phase 12-case benchmark, session protection, statistics and statuses |
| `tools/v2/profile.py` | Actual Nsight capture and report export; no fabricated evidence |
| `tools/v2/verify_pair.cpp` | Real-image same-input stage and end-to-end correctness |
| `tools/v2/verify_helpers.h` | Ordered comparisons and homography projection error |
| `tools/v2/s6_reference.cpp` | Untimed CPU mask/canvas/crop diagnostic using V0 helpers |
| `tools/v2/s6_gpu_diagnostics.cu` | Untimed diagnostic replay of actual V1/V2 S6 kernels |
| `tools/v2/test_reporting.py` | Host-only reporting/failure/session regression tests |

Verification S6 translation units include the original production `.cpp/.cu`
once so private helpers/kernels remain observable without editing V1. In the
verification binary these units replace the original S6 objects. They must not
be linked together with those objects. Production benchmark binaries always use
the original S6 source list, without diagnostic work.

## Correctness

Apply the initial team thresholds from PLAN consistently; confirm them using
V0/V1 instead of relaxing V2 alone:

- Same-input gray/blur/Harris use the PLAN absolute/relative tolerances.
- Keypoint count, coordinates and order match; score uses Harris tolerance.
- Descriptor bits and all ordered match fields match. Real-pair verification
  checks both same-input matchers and propagated full pipeline output.
- Inliers after refinement match. Compare H via B-to-A projection on keypoints
  and a grid covering B; maximum error is `1e-3 px`, rather than coefficient error.
- Canvas/origin, valid mask and crop coordinates/dimensions match. RGB error is
  at most one level per channel against V0, with MAE reported too.
- Direct V1/V2 comparisons demand exact matches, projected coordinates and RGB.
- Failure status is recorded explicitly. Equal pipeline failure is not a
  verified panorama and gets no valid speedup; investigate CUDA errors separately.

The mask diagnostics compare actual GPU validity with the CPU geometric reference,
not with nonblack RGB pixels. They replay inherited kernels outside timed runs.

## Run

On a Colab GPU/Linux checkout containing the complete latest source:

```bash
bash tools/v2/build.sh
build/stage34_v2_selftest
build/test_v2_pipeline
compute-sanitizer --tool memcheck --error-exitcode 1 build/stage34_v2_selftest
compute-sanitizer --tool synccheck --error-exitcode 1 build/stage34_v2_selftest
```

Use a NEW run directory for each comparison:

```bash
python tools/v2/benchmark.py --phase baseline --run-dir reports/v2_benchmark/run01
python tools/v2/profile.py --run-dir reports/v2_benchmark/run01 --version 1
python tools/v2/benchmark.py --phase optimized --run-dir reports/v2_benchmark/run01
python tools/v2/profile.py --run-dir reports/v2_benchmark/run01 --version 2
```

The notebook creates a unique run ID automatically. Baseline refuses to overwrite
an old run. Optimized/profiling phases require the same GPU UUID, OS session,
compiler, source, binaries, commit and dataset hash. Working-tree status is saved
but creating report files alone does not invalidate a session.

Each case/version has one warm-up and five measured runs. Reports preserve all
samples and means; Python derives minimum/maximum, standard deviation and CV,
including stage totals. Each result prints its GPU. Failed cases keep logs and
statuses, not successful speedup ratios. Save the executed notebook, complete
run directory, raw profiler reports, tables and panoramas before Colab resets.

## Verification available locally

Host tests check reporting statistics, sample validation, failure preservation
and session mismatch rejection:

```bash
python -m unittest discover -s tools/v2 -p test_benchmark.py
```

Python syntax and Bash syntax can be checked locally. CUDA compilation, parity,
memory/barrier safety and actual performance still require a CUDA GPU run.

## CLI documentation consulted

- [NVIDIA Nsight Systems User Guide](https://docs.nvidia.com/nsight-systems/UserGuide/index.html)
- [NVIDIA Nsight Systems Analysis Guide](https://docs.nvidia.com/nsight-systems/AnalysisGuide/)
- [NVIDIA Nsight Compute CLI](https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html)

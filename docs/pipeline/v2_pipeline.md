# V2 — S4 shared-memory descriptor tiling

Owner: Khánh. Implementation and notebook preparation assisted by OpenAI Codex.
Source: the team's V0/V1 implementations and the Final Project specification.

## Scope

V2 changes only candidate descriptor access in S4. Each block has 128 threads,
one query per thread, and cooperatively loads 128 descriptors into 4 KB shared
memory. Both matching directions retain ascending traversal, Hamming distance,
top-2, ratio test, cross-check, tie-break and ordered output. Inactive query
threads still reach both barriers; partial tiles only read valid candidates.

S1–S3, S5–S6 and intermediate host/device transfers inherit V1 unchanged.
There is no new S6 implementation or separate S6 diagnostic suite in V2.
The common CLI exposes versions 0, 1 and 2; V2 is enabled with
`PANORAMA_WITH_CUDA` and `PANORAMA_WITH_V2`.

## Notebook structure, following the specification

After setup/build and quick S4/integration checks, `test_v2.ipynb` has five parts:

1. **Bottleneck:** V1 per-stage timing, kernel/copy/end-to-end and real profiler
   output supporting S4 as the optimization target.
2. **Idea:** explain candidate reuse and synchronization/cache tradeoffs.
3. **Implementation:** actual kernel excerpt, thread/block mapping, partial tiles
   and matching invariants; describe the single change from V1.
4. **Result:** verify against V0 and V1, then report S4 and end-to-end timing and
   speedup on all 12 cases. Full stage/kernel/copy timing remains in CSV and a
   collapsible notebook table. Show one figure with two panels: S4 and end-to-end
   speedup, each against V0 and V1. Gallery shows V2 Medium outputs at the three
   sites and the challenging pair; all full-resolution outputs are saved.
5. **Analysis:** discuss faster/slower cases, S4 versus end-to-end improvement,
   correctness failures and limitations using measured evidence.

The group's final comparison across all versions and best-version breakdown
belong in the final report section; V2 does not repeat those charts. Profiler
V2 is available on demand but not required by this notebook's main workflow.
No separate stage tests for inherited S1–S3/S5/S6, statistical CV tables, source
hashes or OS/session identity gates are added to the V2 experiment.

## Files

| File | Purpose |
|---|---|
| `src/stage_implementations/v2_stage4.cu` | Tiled S4 matcher |
| `src/core_headers/v2_stages.h` | Matcher declaration |
| `src/cuda/v2_pipeline.cu`, `.h` | Integrate S4 V2 with inherited V1 stages |
| `tools/v2/build.sh` | Build all experiment binaries with common flags |
| `tools/v2/test_matching.cpp` | Exact S4 parity, tile/tie/ratio/cross-check cases |
| `tools/v2/test_pipeline.cpp` | Basic synthetic integration and direct V1/V2 comparison |
| `tools/v2/verify_pair.cpp` | Same-input S4 and full-pipeline output verification on real pairs |
| `tools/v2/verify_helpers.h` | Ordered comparison and homography projection error |
| `tools/v2/benchmark_pair.cpp` | Warm-up, timed production pipeline runs, samples/mean and panorama |
| `tools/v2/benchmark.py` | Run 12 cases, preserve failures, generate timing/minimum/speedup tables |
| `tools/v2/profile.py` | Collect actual Nsight evidence separately from benchmark timing |
| `tools/v2/test_reporting.py` | Development tests for report correctness, outside notebook workflow |

## Verification

The specification requires results consistent with V0; it does not prescribe
numeric tolerances. Current group criteria compare ordered matches exactly,
keypoints and inliers, H using projection error on points/a grid in B (maximum
`1e-3 px`) and panorama shape/pixels (maximum one RGB level versus V0, plus MAE).
Direct V1/V2 matches, projected coordinates and panorama must agree exactly.
These are acceptance choices, not measurements or thresholds stated by the PDF.

V1 extraction supplies identical descriptor inputs to V0/V1/V2 matchers. The
verifier then compares real full-pipeline outputs; it does not independently
re-test gray/blur/Harris/BRIEF or access private S6 masks/crop internals. This is
the agreed reduced V2 scope; the stricter internal S6 checks in the team PLAN
are not implemented in this notebook.

Failure statuses and logs remain visible. Equal failure is not panorama
correctness PASS; failed or unverified results do not receive valid speedup.

## Run on Colab GPU

Use the latest complete V2 checkout, then run `test_v2.ipynb` top to bottom.
Equivalent commands from the repository root:

```bash
bash tools/v2/build.sh
build/stage34_v2_selftest
build/test_v2_pipeline
python tools/v2/benchmark.py --phase baseline --run-dir reports/v2_benchmark/run01
python tools/v2/profile.py --run-dir reports/v2_benchmark/run01 --version 1
python tools/v2/benchmark.py --phase optimized --run-dir reports/v2_benchmark/run01
```

Use a fresh result directory and keep the same Colab GPU session for all versions.
The notebook creates its directory automatically. One warm-up and five timed
runs per case/version produce means, raw samples and minimum times. Kernel/copy
timing uses pipeline CUDA events; end-to-end includes overhead/transfers and
excludes image loading/saving. Each result prints the GPU name. Do not mix
profiler timings into speedup calculations. Missing profiler support is reported
honestly and remains missing evidence rather than a claimed PASS.

Save the executed notebook, raw JSON/logs, `summary.csv`, `stage_comparison.csv`,
panoramas and profiler reports. Analysis must reference actual cell outputs.

## Local checks and references

```bash
python -m unittest discover -s tools/v2 -p test_reporting.py
```

Host tests verify reporting only. CUDA build, parity and performance require GPU
execution; no such result is fabricated in the notebook.

- Final Project PDF: sections 3, 4.2 and 5.2.
- [NVIDIA Nsight Systems User Guide](https://docs.nvidia.com/nsight-systems/UserGuide/index.html)
- [NVIDIA Nsight Systems Analysis Guide](https://docs.nvidia.com/nsight-systems/AnalysisGuide/)
- [NVIDIA Nsight Compute CLI](https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html)

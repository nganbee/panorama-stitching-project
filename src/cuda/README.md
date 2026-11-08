# CUDA implementation and verification status

Authors: Team 8, with Codex implementation assistance. This code is an original
implementation of the same algorithms used by the project V0. No external source
code was copied. `common.h` remains the source of all pipeline parameters.

**The CUDA source is implemented; CUDA compilation, device correctness and GPU
performance must still be verified on a computer with the NVIDIA CUDA toolkit
and an NVIDIA GPU.** No GPU benchmark numbers are supplied or inferred from CPU
measurements. A successful CPU build does not validate this optional backend.

## Six stages and three versions

The full pipeline calls actual device kernels for all six stages; it does not
call the sequential stages as a GPU implementation. Image loading/saving is
outside `run_v1`, `run_v2` and `run_v3`, as it is for V0.

| Workload | V1: direct CUDA baseline | V2: memory/warp optimization | V3: additional RANSAC/resource optimization |
|---|---|---|---|
| Grayscale | One thread per RGB pixel | Same | Same |
| Separable Gaussian | Global-memory reads, 16x16 output blocks | Cooperative shared-memory halo tiles for horizontal/vertical passes; fixed coefficients in constant memory | Same as V2 |
| Harris | Sobel tensor products, then global-memory 5x5 window sum | Tensor products plus shared 20x20 halo tiles for 16x16 output windows | Same as V2 |
| NMS | Two-pass device maximum reduction, one thread per pixel, atomic candidate compaction | Same | Same |
| BRIEF | One thread computes a complete descriptor | One warp per keypoint; two warp ballots construct each 64-bit word | Same as V2 |
| Hamming matching | One thread per query scans all candidates, `__popcll`, both directions | Blocks share 128 candidate descriptors while each query remains in registers | Same as V2 |
| RANSAC | One thread solves each normalized four-point model; one thread serially scores each hypothesis | Same solver/scoring as V1 | One block per hypothesis scores matches in parallel and tree-reduces count/error |
| Inverse warp and feather | One thread per output pixel; double-precision inverse coordinates and bilinear interpolation | Same | Same |
| Allocation | RAII scratch allocated per pipeline call | Same | Per-thread scratch and event objects retained across calls; buffers grow only when needed |

These are explicit optimization hypotheses, not a claim that V2/V3 always run
faster. Shared memory may reduce occupancy, warp-per-keypoint BRIEF can waste
lanes at small feature counts, and double-precision RANSAC performance depends
on the device. The final report must justify the measured speedups using actual
profiles; hot stages must be chosen from the measured V0 timing table.

BRIEF uses the exact xorshift32 seed/draw order and bit layout of
`v0_stage34.cpp`. Keypoint ranking uses response descending, y ascending, x
ascending. Matching keeps the first index at tied distances, requires a second
neighbour, rejects a zero-distance ambiguity and applies the same ratio test
and mutual-nearest-neighbour cross-check. The homography maps **B into A**.
The normalized 8-parameter least-squares solver uses doubles and partial pivoting.
Sampling uses deterministic per-hypothesis xorshift32 streams seeded from
`kRansacSeed` on the device. This keeps V1/V2/V3 on the same sample sequence;
the sequence differs from V0's host MT19937 sampler. V3 sums scoring errors in
a different reduction order, so tied or nearly tied hypotheses may choose a
different equally valid result. Check final reprojection accuracy, inliers and
panorama tolerance rather than requiring bit-identical homographies.

The warp uses inclusive pixel-center bounds, scale-relative projective guards,
boundary tolerance, double interpolation and one final rounded RGB conversion.
When both border distances are zero, each valid contributor gets weight 0.5.
Crop validity comes from the geometric coverage mask, so real black pixels are
not discarded. Invalid, singular, horizon-crossing or unbounded transforms fail.
The canvas is capped at 32768 per dimension and 64M pixels. Device input images
also have these limits to keep kernel indices and allocations bounded.

## Explicit host phases

The following phases remain on the CPU, are deliberate and are included in their
corresponding stage time and in end-to-end time:

* NMS: deterministic candidate partial-sort/top-K after device compaction. The
  sorted keypoints are uploaded for BRIEF.
* Matching: raster-order compaction of accepted device match records.
* RANSAC: the host uploads keypoints and accepted matches, then downloads only
  the final homography and inlier count. Correspondence filtering, seeded
  four-index sampling, hypothesis estimation/scoring, best-model selection,
  normalized least-squares refinement and final inlier counting run on device.
* Warp: four-corner canvas validation/bounds and the linear-time largest
  axis-aligned rectangle crop of the downloaded device coverage mask.

No source image, blurred image or descriptor array is transferred between CPU
stages during ordinary pipeline execution. V1/V2/V3 all retain those arrays on
the device. Optional debug keypoints/matches reuse the host control records.

## Timing interpretation

`gray`, `blur`, `harris`, `nms_topk`, `describe`, `match`, `ransac` and
`warp_blend` contain CUDA event intervals plus the explicitly timed host phases
above. The first five fields sum the work for **both A and B**. Each kernel group
is synchronized at its ending event, so these measurements are reproducible but
do not measure an overlapping asynchronous pipeline.

`kernel_total` sums only CUDA event intervals surrounding kernels. `h2d` and
`d2h` have separate event intervals around actual copies, including constant
tables and control records. Device allocation, event creation, host vector
allocation, scalar setup and device memset overhead can remain outside these
stage/transfer sums. `end_to_end` uses `steady_clock` around the entire call,
including allocation, control work, synchronization and final crop, while
excluding load/save. Thus stage sums and device-event totals need not equal the
end-to-end time. V3 buffer reuse affects this last measurement after warm-up.
Use the same warm-up count, repetitions and image pairs for all versions.

## Build and run on a CUDA machine

From the project root, use the optional CMake backend:

```powershell
cmake -S . -B build-cuda -DPANORAMA_WITH_CUDA=ON
cmake --build build-cuda --config Release
```

`src/cuda/cuda_pipeline.cu` is the only CUDA translation unit. The build must
define `PANORAMA_WITH_CUDA` for the version registry and compile with C++17.
Retain `--fmad=false` and avoid `--use_fast_math` when comparing to V0. Consult
the root README for the executable path and command-line arguments for your
generator/platform. V0 remains available in a CPU-only build; GPU requests are
rejected if that backend was not compiled, rather than falling back silently.

The Python build helper also supports `python scripts/build.py --cuda` on a
CUDA machine. The executable validation commands are:

```powershell
stitch --selftest --version 1
stitch --selftest --version 2
stitch --selftest --version 3
```

Use the actual built executable path in place of `stitch`. The CUDA CMake build
registers these device stage tests and the end-to-end synthetic tests with CTest.

`cuda_selftest(version)` is the device validation entry point for each of V1,
V2 and V3. It exercises partial edge tiles (97x83), V0 grayscale/blur/Harris
tolerances, exact NMS/top-K and BRIEF on identical input arrays, duplicate
descriptor ratio/cross-check cases, a known translation with forty inliers and
eight outliers, and identity/integer translation/perspective bilinear feather
and crop parity. It returns false on a CUDA runtime error or any failed check.
The `host_math_selftest.cpp` test can run without CUDA and checks only the shared
mathematical utility; it does not substitute for this device validation.

Before reporting GPU results, run every device selftest, compare end-to-end
synthetic homography metrics to ground truth, compare the real-image outputs to
V0, and run NVIDIA Compute Sanitizer on the selftests and edge image sizes.
Then collect actual stage/transfer/end-to-end timings at every required image
resolution. Record GPU model, toolkit/compiler/driver versions and build flags.
Any numerical or runtime failures must be fixed before claiming CUDA completion.

## Remaining group-owned work

Enter the actual contributors/student IDs, approve the final group task table,
provide the required real-location image pairs and download link, and record
actual GPU validation, profiling, timing tables and final discussion. The
synthetic fixtures and implementation plan are support material, not evidence
that the real-location dataset or GPU results have been completed.

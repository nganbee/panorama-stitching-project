# CUDA source status and device validation

CUDA V1–V3 sources exist, with actual device kernels for all six stages. They have not been compiled or executed on a CUDA device in the CPU verification session. See `src/cuda/README.md` for exact source mapping, numerical conventions and host phases. No GPU speedup is claimed before device measurement.

V1 maps grayscale/blur/Harris/NMS/warp to pixels, BRIEF to one keypoint per thread, matching to one query per thread and RANSAC model/scoring to a hypothesis per thread. V2 adds shared-memory halo filters/Harris accumulation, warp-per-keypoint BRIEF and tiled descriptor matching. V3 retains V2 and adds block-parallel RANSAC scoring/reduction plus scratch/event reuse across calls.

The following work is deliberately on the CPU and included in the relevant stage/end-to-end time: deterministic candidate top-K, match compaction, seeded sample generation/model selection and inlier refinement, canvas validation and largest-covered-rectangle crop. These CPU phases are not evidence that GPU kernels accelerate the entire stage infinitely. Amdahl based on whole V0 stage sums is an ideal upper bound; partial host phases and transfers reduce realized speedup.

```bash
python scripts/build.py --cuda
build/stitch --selftest --version 1
build/stitch --selftest --version 2
build/stitch --selftest --version 3
build/synthetic_pipeline_test out/synthetic_v1 --version 1
build/synthetic_pipeline_test out/synthetic_v2 --version 2
build/synthetic_pipeline_test out/synthetic_v3 --version 3
python scripts/benchmark.py out/prepared/manifest.prepared.json --executable build/stitch --runs 5 --versions 0 1 2 3 --output out/gpu_benchmark
```

Use `.exe` on Windows and nvcc's supported host compiler. An unsupported version, unavailable backend, runtime error or failed correctness test is an error; do not fall back to V0 while labelling it CUDA. Device selftests compare stage outputs to V0 (partial tiles, NMS ties, exact BRIEF, matching ambiguity, homography/outliers and warp/crop parity). End-to-end synthetic tests check known translation/projective corner errors. CPU host-math selftests check only shared helper math and do not establish device correctness.

Run Compute Sanitizer on device selftests and synthetic edge cases before using benchmark numbers. Compare real panorama quality to V0, then record GPU model/driver/toolkit/compiler/build flags and run all versions with the same inputs/session/warm-up/runs. Record failure cases rather than filtering them from the report.

`kernel_total`, `h2d`, `d2h` use CUDA event intervals. Stage fields include associated kernel intervals and explicit host work; end-to-end includes allocation, control, synchronization and crop while excluding image load/save. Totals need not be equal. V3 reuse primarily changes end-to-end after warm-up. Nsight Systems should explain transfers/launch/control overhead; Nsight Compute should support claims about memory bandwidth, occupancy and reduction/tiling. Final performance claims, plots and optimization analysis must use this recorded device session, not synthetic CPU extrapolation.

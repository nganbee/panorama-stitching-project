# V2 — S4 shared-memory descriptor tiling

Owner: Khánh. Implementation assisted by OpenAI Codex; derived from the team's V1.

## Files

- `src/stage_implementations/v2_stage4.cu`: independent `v2_match` implementation.
- `src/core_headers/v2_stages.h`: S4 interface.
- `src/cuda/v2_pipeline.cu` and `.h`: independent V2 pipeline entry point.
- `tools/run_stage34_v2_cuda_selftest.cpp`: compares V0/V1/V2 matching, including
  partial tiles, 127/128/129 boundaries, ties, empty inputs and 3000 keypoints.
- `tools/test_v2_pipeline.cpp`: synthetic V0/V1/V2 pipeline correctness and timing.
- `test_v2.ipynb`: GPU tests, 12-case benchmark and comparison.

V1 implementation and tests remain unchanged. V2 reuses V1 S1–S3, S5 and S6.
The registry exposes V2 only with `PANORAMA_WITH_CUDA` and `PANORAMA_WITH_V2`,
so existing V1 build commands do not require linking V2 sources.

## Optimization

Each block has 128 threads, one query per thread. Threads cooperatively load
128 candidate descriptors (4096 bytes) into shared memory, then each active
query scans the tile in ascending candidate order. Both directions use tiling.
Inactive queries in the final block still reach both barriers. The last tile
is bounded by the candidate count, so uninitialized slots are never read.

Hamming XOR/popcount, descriptor layout, lowest-index tie-break, strict ratio
test, mutual cross-check and ascending output order remain identical to V1.
Filtering and single-thread GPU compaction are inherited unchanged. Host-buffer
copies and allocations are also unchanged; device residency belongs to V5.

Candidate loads can be reused by up to 128 queries. Barrier overhead and V1's
existing cache reuse may limit or reverse speedup, especially for small inputs.
Timing V1 must justify S4 as the target; no speedup is claimed before measurement.

## Run on Colab

Use the current V2 source checkout, or upload an archive containing `src/`,
`tools/`, `dataset/` and `test_v2.ipynb`. A notebook alone does not upload source.
From the repository root:

```bash
bash tools/build_v2.sh
build/stage34_v2_selftest
build/test_v2_pipeline
compute-sanitizer --error-exitcode 1 build/stage34_v2_selftest
compute-sanitizer --tool synccheck --error-exitcode 1 build/stage34_v2_selftest
build/stitch_v2 dataset/location1/small_720p/1.jpg dataset/location1/small_720p/2.jpg out/v2.png --version 2 --runs 5 --json reports/v2.json
```

`test_v2.ipynb` runs all three versions on the same GPU, with one warm-up and
five timed runs per case. It saves panorama/JSON output, compares V2 with V1
and V0, reports failures explicitly and computes speedup only for valid cases.

## Report checklist

1. Bottleneck: save actual V1 timing/profiling showing S4's contribution.
2. Idea: explain descriptor reuse and the cost of barriers.
3. Implementation: show the tile kernel, mapping, and preserved matching rules.
4. Result: save parity, sanitizer and all 12-case timing/correctness outputs.
5. Analysis: compare S4 and end-to-end speedup against both V0 and V1, including
   slower cases. All numbers must come from executed notebook outputs.

Local CUDA build and GPU execution have not been verified in this environment.

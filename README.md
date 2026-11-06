# Panorama stitching project

## V1 CUDA S3-S4 (global-memory baseline)

The independent GPU implementation lives in
`src/stage_implementations/v1_stage34.cu`, with interfaces in
`src/core_headers/v1_stage34.h`. Call `v1_describe` for each blurred image
and its keypoints, then `v1_match` on the two descriptor vectors. The host
wrappers allocate buffers, copy data, and launch kernels. BRIEF pattern
generation, descriptor calculation, nearest-neighbour searches, ratio test,
mutual cross-check, and ordered match compaction all run on the GPU.
CUDA errors throw exceptions; there is no CPU fallback.

This is only the S3-S4 contribution, not a complete `--version 1` pipeline.
Keep V0 as the CPU reference. S1-S2 and S5-S6 must have their own GPU
implementations before enabling the CUDA pipeline registry. Intermediate
host buffers are supported by these interfaces. Calls are synchronous;
wall-clock measurements include allocation and transfers, not just kernels.
Compaction deliberately uses one GPU thread for this simple baseline.

Build and run the parity test on a machine with the CUDA toolkit and GPU
(the CPU functions here are used only as a test oracle):

```bash
nvcc -O2 -std=c++17 tools/run_stage34_cuda_selftest.cpp src/stage_implementations/v0_stage34.cpp src/stage_implementations/v1_stage34.cu -o stage34_cuda_selftest
./stage34_cuda_selftest
compute-sanitizer ./stage34_cuda_selftest
```

The test checks all descriptor bits and ordered match fields against V0,
including borders, fractional coordinates, invalid keypoints, empty inputs,
single candidates, duplicate descriptors, and unequal descriptor counts.

## V0 build

The V0 implementation is a single-threaded C++17 pipeline. From the project root:

```bash
g++ -O2 -std=c++17 \
  -Isrc/core_headers -Isrc/libraries \
  src/integration/main.cpp \
  src/integration/v0_pipeline.cpp \
  src/stage_implementations/v0_stage12.cpp \
  src/stage_implementations/v0_stage34.cpp \
  src/stage_implementations/v0_stage5.cpp \
  src/stage_implementations/v0_stage6.cpp \
  src/libraries/stb_impl.cpp \
  -o stitch
```

Run the pipeline with one warm-up run and at least five timed runs:

```bash
./stitch data/left.png data/right.png out/panorama.png --version 0 --runs 5
```

The reported timings exclude image loading and saving. Stage 1-3 timings are
the sum over both input images. Stage 5 estimates a homography mapping image B
to image A, which is the coordinate convention required by Stage 6.

Stage 6 builds a valid-pixel mask while warping and crops the panorama to the
largest fully-covered axis-aligned rectangle, removing black empty triangles
caused by perspective warping. To inspect a difficult pair, request optional
debug artifacts:

```bash
./stitch data/left.jpg data/right.jpg out/panorama.png --version 0 --runs 5 \
  --debug-prefix out/debug_left_right
```

This writes the normalized homography matrix, keypoint overlays for both
images, and a side-by-side match image. Match lines are green for reprojection
inliers and red for rejected matches. Debug artifact generation is outside the
timed pipeline.

## S1-S2 debug notebook

`testkeypoints.ipynb` builds `tools/debug_keypoints.cpp` using the current
source layout and produces grayscale, blurred, and keypoint-overlay images in
`out/`.

## S1-S2 timing harness

The harness performs one warm-up per resolution and averages at least five
timed runs over both images. Passing a left/right pair resizes the same scene
to all three required resolutions; omitting the pair uses deterministic
synthetic images for a quick scaling check:

```bash
g++ -O2 -std=c++17 -Isrc/core_headers -Isrc/libraries \
  tools/timing_stage12.cpp src/stage_implementations/v0_stage12.cpp \
  src/libraries/stb_impl.cpp \
  -o timing_stage12
./timing_stage12 5 data/left.png data/right.png
```

It reports 1280x720, 1920x1080, and 3840x2160 results in CSV-like lines.

## Synthetic S5 check

The homography self-test uses a known translation and injected outlier matches:

```bash
g++ -O2 -std=c++17 -Isrc/core_headers \
  tools/synthetic_homography_test.cpp src/stage_implementations/v0_stage5.cpp \
  -o synthetic_homography_test
./synthetic_homography_test
```

The independent S1-S2 tests can be run with:

```bash
g++ -O2 -std=c++17 -Isrc/core_headers \
  tools/run_stage12_selftest.cpp src/integration/v0_selftest.cpp \
  src/stage_implementations/v0_stage12.cpp \
  -o stage12_selftest
./stage12_selftest
```

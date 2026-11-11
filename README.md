# Panorama stitching project

## Chạy project trên Google Colab

### 1. Bật GPU runtime

Trong Colab chọn:

```text
Runtime -> Change runtime type -> Hardware accelerator -> T4 GPU
```

Sau đó kiểm tra GPU và CUDA:

```python
!nvidia-smi
!nvcc --version
```

### 2. Clone repository và chuyển sang branch V1

```python
!git clone https://github.com/nganbee/panorama-stitching-project.git
%cd panorama-stitching-project
!git checkout v1
!mkdir -p build out reports
```

Nếu repository đã được clone ở cell trước đó, chỉ cần chạy:

```python
%cd /content/panorama-stitching-project
!git pull origin v1
```

### 3. Build và chạy V0 CPU reference

V0 dùng để kiểm tra correctness và làm baseline thời gian cho V1:

```python
!g++ -O2 -std=c++17 \
  -Isrc/core_headers -Isrc/libraries \
  src/integration/main.cpp \
  src/integration/v0_pipeline.cpp \
  src/stage_implementations/v0_stage12.cpp \
  src/stage_implementations/v0_stage34.cpp \
  src/stage_implementations/v0_stage5.cpp \
  src/stage_implementations/v0_stage6.cpp \
  src/libraries/stb_impl.cpp \
  -o build/stitch
```

Chạy thử với cặp ảnh 720p:

```python
!build/stitch \
  dataset/location1/small_720p/1.jpg \
  dataset/location1/small_720p/2.jpg \
  out/panorama_v0_location1_720p.png \
  --version 0 --runs 5 \
  --json reports/v0_location1_720p.json
```

Hiển thị panorama trong Colab:

```python
from IPython.display import Image, display
display(Image(filename="out/panorama_v0_location1_720p.png"))
```

### 4. Chạy self-test S1–S2 CUDA của V1

Milestone 1 kiểm tra grayscale, Gaussian blur, Harris response, NMS và
top-K của V1 so với V0:

```python
!nvcc -O2 -std=c++17 -Isrc/core_headers \
  tools/run_stage12_cuda_selftest.cpp \
  src/stage_implementations/v0_stage12.cpp \
  src/stage_implementations/v1_stage12.cu \
  -o build/stage12_cuda_selftest

!build/stage12_cuda_selftest
```

Kiểm tra thêm lỗi memory access bằng Compute Sanitizer:

```python
!compute-sanitizer build/stage12_cuda_selftest
```

Kết quả thành công cần có:

```text
S1/S2 CUDA parity tests passed
```

### 5. Chạy self-test S3–S4 CUDA

```python
!nvcc -O2 -std=c++17 -Isrc/core_headers \
  tools/run_stage34_cuda_selftest.cpp \
  src/stage_implementations/v0_stage34.cpp \
  src/stage_implementations/v1_stage34.cu \
  -o build/stage34_cuda_selftest

!build/stage34_cuda_selftest
```

Kết quả thành công cần có:

```text
S3/S4 CUDA parity tests passed
```

### 6. Chạy self-test S6 CUDA

```python
!nvcc -O2 -std=c++17 -Isrc/core_headers \
  tools/test_v1_stage6.cpp \
  src/stage_implementations/v0_stage6.cpp \
  src/stage_implementations/v1_stage6.cu \
  -o build/test_v1_stage6

!build/test_v1_stage6
```

Kết quả thành công cần có dòng:

```text
Result: ALL PASS
```

### 7. Đo timing V0 của S1–S2

Timing harness thực hiện một warm-up và trung bình ít nhất năm lần chạy ở
720p, 1080p và 4K:

```python
!g++ -O2 -std=c++17 -Isrc/core_headers -Isrc/libraries \
  tools/timing_stage12.cpp \
  src/stage_implementations/v0_stage12.cpp \
  src/libraries/stb_impl.cpp \
  -o build/timing_stage12

!build/timing_stage12 5 \
  dataset/location1/small_720p/1.jpg \
  dataset/location1/small_720p/2.jpg
```

### Trạng thái V1 hiện tại

Trên branch `v1`, các thành phần CUDA đã có self-test độc lập cho S1–S2,
S3–S4 và S6. Full pipeline `--version 1` chưa được bật trên branch này vì
phần `src/cuda/cuda_pipeline.*` chưa được merge vào `v1`. Do đó, trước mắt
hãy dùng V0 để tạo baseline và chạy các parity test CUDA theo từng stage.

Toàn bộ quy trình Colab ở trên đã được đóng gói trong notebook
`test_v1.ipynb`. Có thể mở notebook này trực tiếp trên Google Colab hoặc
upload file lên Colab rồi chạy các cell theo thứ tự.

## V1 CUDA S1-S4 (basic stage baselines)

The basic GPU implementation of S1-S2 lives in
`src/stage_implementations/v1_stage12.cu`, with blocking host-buffer
interfaces in `src/core_headers/v1_stages.h`. Grayscale, both Gaussian blur
passes, Sobel/tensor construction, Harris response, and NMS run on the GPU.
For this first V1 milestone, deterministic top-K selection remains on the
host after the GPU NMS candidate list is copied back. Shared-memory tiling,
kernel fusion, device-side top-K, and persistent workspaces are reserved for
later versions.

Build and run the S1-S2 parity test on a machine with the CUDA toolkit and GPU:

```bash
nvcc -O2 -std=c++17 -Isrc/core_headers \
  tools/run_stage12_cuda_selftest.cpp \
  src/stage_implementations/v0_stage12.cpp \
  src/stage_implementations/v1_stage12.cu \
  -o stage12_cuda_selftest
./stage12_cuda_selftest
compute-sanitizer ./stage12_cuda_selftest
```

The test compares V1 against V0 for grayscale, blur, Harris response, and
keypoint count/order on deterministic random and constant images.

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

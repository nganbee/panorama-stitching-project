# Panorama Stitching Project (CUDA)

Đồ án song song hoá pipeline ghép ảnh panorama (S1–S6) bằng CUDA trên GPU.

## Các version đã thiết lập

| Version | Tối ưu chính | Source chính | Notebook kiểm chứng |
|---|---|---|---|
| **V0** | CPU tuần tự (C++17), chuẩn correctness & baseline | `src/integration/v0_pipeline.cpp`, `src/stage_implementations/v0_*.cpp` | `test_v0.ipynb` |
| **V1** | Basic CUDA toàn bộ S1–S6 trên GPU | `src/cuda/cuda_pipeline.cu`, `src/stage_implementations/v1_*.cu` | `test_v1.ipynb` |
| **V2** | S4 shared-memory descriptor tiling | `src/cuda/v2_pipeline.cu`, `src/stage_implementations/v2_stage4.cu` | `test_v2.ipynb` |
| **V3** | S6 compressed monotonic-stack cho crop | `src/cuda/v3_pipeline.cu`, `src/stage_implementations/v3_stage6.cu` | `test_v3.ipynb` |
| **V4** | Tối ưu S6 `warp_blend` | `src/cuda/v4_pipeline.cu`, `src/stage_implementations/v4_stage6.cu` | `test_v4_1.ipynb` |
| **V5** | GPU Data Residency (giữ buffer trung gian trên GPU, kế thừa kernel V4) | `src/cuda/v5_pipeline.cu`, `src/stage_implementations/v5_stage4.cu`, `src/stage_implementations/v5_stage6.cu` | `test_v5.ipynb` |
| **V7** | S5 RANSAC một warp / hypothesis (ballot + popc đếm inlier, chỉ đổi S5; hiện build trên V5 vì V6 chưa ghép) | `src/cuda/v7_pipeline.cu`, `src/stage_implementations/v7_stage5.cu` | `test_v7.ipynb` |

Lộ trình mở rộng được ghi trong `docs/PLAN.md`: **V6** (owner Khánh, chưa ghép) và **V7** (S5 RANSAC một warp / hypothesis, xem `docs/pipeline/v7_pipeline.md`).

## Cấu trúc thư mục

- `src/core_headers/`: tham số chung (`common.h`), interface từng stage (`v*_stages.h`), registry (`versions.h`), JSON I/O (`report_io.h`).
- `src/stage_implementations/`: cài đặt S1–S6 cho từng version (`v0_*` → `v5_*`, `v7_stage5.cu`).
- `src/cuda/`: điều phối pipeline CUDA (`cuda_pipeline.cu`, `v2_pipeline.cu` → `v5_pipeline.cu`, `v7_pipeline.cu`).
- `src/integration/`: CLI entrypoint (`main.cpp`) và `v0_pipeline.cpp`.
- `src/tools/`: unit/parity test, benchmark và profiler theo từng version (`v2/`, `v3/`, `v4/`, `v5/`, `v7/`).
- `docs/`: `PLAN.md` và tài liệu thiết kế từng version.

## Build & chạy nhanh (Linux / Google Colab GPU)

```bash
mkdir -p build out
nvcc -O2 -std=c++17 --fmad=false \
  -DPANORAMA_WITH_CUDA -DPANORAMA_WITH_V2 -DPANORAMA_WITH_V3 -DPANORAMA_WITH_V4 -DPANORAMA_WITH_V5 -DPANORAMA_WITH_V7 \
  -Isrc/core_headers -Isrc/libraries \
  src/integration/main.cpp src/integration/v0_pipeline.cpp \
  src/stage_implementations/v0_stage12.cpp src/stage_implementations/v0_stage34.cpp \
  src/stage_implementations/v0_stage5.cpp src/stage_implementations/v0_stage6.cpp \
  src/stage_implementations/v1_stage12.cu src/stage_implementations/v1_stage34.cu \
  src/stage_implementations/v1_stage5.cu src/stage_implementations/v1_stage6.cu \
  src/stage_implementations/v2_stage4.cu src/stage_implementations/v3_stage6.cu \
  src/stage_implementations/v4_stage6.cu src/stage_implementations/v5_stage4.cu \
  src/stage_implementations/v5_stage6.cu src/stage_implementations/v7_stage5.cu \
  src/cuda/cuda_pipeline.cu src/cuda/v2_pipeline.cu src/cuda/v3_pipeline.cu \
  src/cuda/v4_pipeline.cu src/cuda/v5_pipeline.cu src/cuda/v7_pipeline.cu \
  src/libraries/stb_impl.cpp -o build/stitch
```

```bash
./build/stitch --list-versions
./build/stitch --selftest --version 5
./build/stitch --selftest --version 7
./build/stitch dataset/location1/small_720p/1.jpg dataset/location1/small_720p/2.jpg out/pano.png --version 5 --runs 5 --json out/report.json
```

Để chạy đầy đủ parity tests, Compute Sanitizer, benchmark 12 cặp ảnh (720p / 1080p / 4K) và Nsight Systems profiling, mở `test_v5.ipynb` / `test_v7.ipynb` (hoặc notebook của version tương ứng) trên Colab GPU và chọn **Run all**.

# V7 pipeline: S5 RANSAC một warp / hypothesis

Owner: Thụ. Theo [PLAN.md](../PLAN.md): V7 **chỉ đổi cách tính inlier score của S5**,
giữ nguyên dãy mẫu, 2.000 vòng, threshold và bước refinement; đo riêng S5 và
end-to-end. Số inlier, homography và panorama phải nhất quán V0.

## 1. Nền code và giới hạn hiện tại

PLAN đặt V7 kế thừa V6 (owner Khánh). Tại thời điểm viết, **V6 chưa được ghép** vào
`main`, nên V7 được xây trên **version mới nhất đã ghép là V5** (V1 S1–S3, V2 S4 tiling,
V3 S6 compressed crop, V4 S6 warp, GPU data residency). PLAN cho phép: "Có S5 V1; có
thể thử trên nhánh riêng trước khi V6 hoàn thành".

`run_v7` gọi đúng các hàm `v5::*` cho S1–S4 và S6, chỉ thay `v5::ransac_homography`
bằng `v7::ransac_homography`. Khi V6 ghép xong, chỉ cần đổi lời gọi stage mà V6 tối ưu
trong [v7_pipeline.cu](../../src/cuda/v7_pipeline.cu) sang hàm V6; S5 không đổi. Sau đó
chạy lại correctness và timing trước khi chốt V7.

So sánh hợp lệ của V7 hiện tại là **V5 → V7** (mọi stage khác giống hệt, chỉ khác S5).

## 2. Bottleneck và lý do chọn

- Nsight V5 (4K): S5 chiếm ~25.2% tổng kernel, đứng #1 ở một số case 720p
  (`refine_best` 13.3%, `generate_samples` 8.1%, `score_hypotheses` 3.8%).
- V1/V5 `score_hypotheses_kernel` dùng **một thread / hypothesis**: chỉ có 2.000 thread
  (~63 warp) cho cả GPU, mỗi thread lặp tuần tự qua n match (n ≤ 3.000). GPU bị dùng
  rất ít (T4 có 40 SM, mỗi SM chứa tới 32 warp), và độ trễ của phép chia double không
  được che.
- `refine_best_kernel` chạy **một thread** cho: duyệt 2.000 candidate, chọn inlier của
  H tốt nhất (n phép chiếu) và đếm inlier của H refined (n phép chiếu nữa).

## 3. Thay đổi so với V5

| Phần S5 | V1/V5 | V7 |
|---|---|---|
| Validate / compact match, sinh mẫu MT19937 | Kernel V1 | **Giữ nguyên** (copy nguyên văn vì kernel V1 là private trong file) |
| DLT 4 điểm cho mỗi hypothesis | Trong `score_hypotheses_kernel`, 1 thread/hypothesis | Tách thành `estimate_hypotheses_v7_kernel`, vẫn 1 thread/hypothesis, cùng công thức |
| Chấm inlier mỗi hypothesis | 1 thread lặp n match | `score_hypotheses_warp_v7_kernel`: **1 warp/hypothesis**, lane l xử lý match `base + l` (đọc coalesced), `__ballot_sync` + `__popc` đếm inlier; 2.000 warp = 64.000 thread |
| Tổng lỗi inlier (tie-break) | `error += e` tuần tự | Cộng **theo đúng thứ tự match**: sau mỗi ballot, duyệt bit từ thấp lên cao và `__shfl_sync` lỗi của lane đó. Cùng các số double, cùng thứ tự → kết quả trùng từng bit |
| Chọn H tốt nhất | 1 thread duyệt 2.000 candidate | Warp argmax: mỗi lane duyệt phần stride của mình, rồi butterfly `__shfl_xor_sync` với thứ tự (nhiều inlier hơn, lỗi nhỏ hơn, iteration sớm hơn) |
| Chọn inlier cho refinement | 1 thread | Warp ballot + prefix `__popc(mask & lanemask_lt)`: ghi `selected[]` **đúng thứ tự** như V1 |
| Least-squares DLT trên mọi inlier | 1 thread | **Giữ nguyên** (lane 0, cùng thứ tự cộng của V0) |
| Đếm inlier của H refined | 1 thread | Warp ballot + `__popc` |

Không đổi: seed và dãy mẫu (kể cả mẫu trùng bị loại), `kRansacIters = 2000`,
`kRansacThresh = 3 px`, công thức lỗi, tie-break, refinement, `--fmad=false`, data path
và số byte copy của V5, cách đo thời gian.

### Vì sao kết quả trùng V5 từng bit

- Số inlier là số nguyên: ballot/popc cho đúng giá trị.
- Tổng lỗi không dùng tree reduction (sẽ đổi thứ tự làm tròn và có thể lật tie-break),
  mà cộng tuần tự theo chỉ số match. Cộng thêm 0 cho lane không phải inlier không xảy
  ra vì chỉ duyệt các bit đã set.
- Argmax dùng thứ tự từ điển có chỉ số iteration duy nhất, nên mọi thứ tự reduction cho
  cùng một phần tử, chính là phần tử mà vòng lặp tuần tự của V1 chọn.
- Compaction có thứ tự nên `selected[]` giống hệt, và DLT refinement chạy y như V1.

Logic này đã được kiểm tra bằng mô phỏng CPU 32 lane (60 bộ dữ liệu ngẫu nhiên, n từ 4
đến 3.000, có outlier và điểm trùng): count, error, best index và `selected[]` trùng V1.

## 4. File

| File | Vai trò |
|---|---|
| [v7_stages.h](../../src/core_headers/v7_stages.h) | Interface `v7::ransac_homography` (cùng chữ ký `v5::ransac_homography`) |
| [v7_stage5.cu](../../src/stage_implementations/v7_stage5.cu) | Kernel V7: `estimate_hypotheses_v7_kernel`, `score_hypotheses_warp_v7_kernel`, `refine_best_warp_v7_kernel`. Dùng lại math S5 của V1 bằng `#include "v1_stage5.cu"` với `PANORAMA_STAGE5_HOST_TEST` (giống test V1); `v1_stage5.cu` không bị sửa |
| [v7_pipeline.cu](../../src/cuda/v7_pipeline.cu) / [.h](../../src/cuda/v7_pipeline.h) | `run_v7` (V5 + S5 V7), `cuda_selftest_v7` (V7 phải trùng V5) |
| `versions.h`, `main.cpp`, `report_io.h` | `PANORAMA_WITH_V7`, `--version 7`, `--selftest --version 7`, `available_versions` |
| [test_v7_pipeline.cpp](../../src/tools/v7/test_v7_pipeline.cpp) | Test S5 riêng (V7 == V5 từng bit, khớp V0) + pipeline (V7 == V5, tiêu chí PLAN với V0) + đo S5 |
| [test_v7.ipynb](../../test_v7.ipynb) | Notebook Colab: build, parity, sanitizer, benchmark 12 case, Nsight |

## 5. Build và chạy trên Colab GPU

```bash
FLAGS="-O2 -std=c++17 --fmad=false -DPANORAMA_WITH_CUDA -DPANORAMA_WITH_V2 -DPANORAMA_WITH_V3 \
  -DPANORAMA_WITH_V4 -DPANORAMA_WITH_V5 -DPANORAMA_WITH_V7 -Isrc/core_headers -Isrc/libraries"
SOURCES="src/integration/v0_pipeline.cpp \
  src/stage_implementations/v0_stage12.cpp src/stage_implementations/v0_stage34.cpp \
  src/stage_implementations/v0_stage5.cpp src/stage_implementations/v0_stage6.cpp \
  src/stage_implementations/v1_stage12.cu src/stage_implementations/v1_stage34.cu \
  src/stage_implementations/v1_stage5.cu src/stage_implementations/v1_stage6.cu \
  src/stage_implementations/v2_stage4.cu src/stage_implementations/v3_stage6.cu \
  src/stage_implementations/v4_stage6.cu src/stage_implementations/v5_stage4.cu \
  src/stage_implementations/v5_stage6.cu src/stage_implementations/v7_stage5.cu \
  src/cuda/cuda_pipeline.cu src/cuda/v2_pipeline.cu src/cuda/v3_pipeline.cu \
  src/cuda/v4_pipeline.cu src/cuda/v5_pipeline.cu src/cuda/v7_pipeline.cu \
  src/libraries/stb_impl.cpp"

nvcc $FLAGS src/integration/main.cpp $SOURCES -o build/stitch_v7
nvcc $FLAGS src/tools/v7/test_v7_pipeline.cpp $SOURCES -o build/test_v7_pipeline

build/stitch_v7 --list-versions            # 0 1 2 3 4 5 7
build/stitch_v7 --selftest --version 7     # V7 == V5 trên cặp synthetic
build/test_v7_pipeline                     # S5 unit + synthetic + ca lỗi
build/test_v7_pipeline dataset/location1/small_720p/1.jpg dataset/location1/small_720p/2.jpg
compute-sanitizer --tool memcheck --error-exitcode 1 build/test_v7_pipeline
build/stitch_v7 dataset/location1/large_4k/1.jpg dataset/location1/large_4k/2.jpg \
  out/pano_v7.png --version 7 --runs 5 --json out/v7_location1_4k.json
```

Đọc `timings_ms.ransac` trong JSON để so S5 giữa V5 và V7; `kernel_total` và
`end_to_end` cho tổng thể. Trong Nsight Systems (`cuda_gpu_kern_sum`), so
`score_hypotheses_kernel` + `refine_best_kernel` (V5) với
`estimate_hypotheses_v7_kernel` + `score_hypotheses_warp_v7_kernel` +
`refine_best_warp_v7_kernel` (V7).

## 6. Rủi ro và hướng tiếp

- DLT 4 điểm vẫn 1 thread/hypothesis; nếu nó chiếm phần lớn thời gian sau V7, đó là
  giới hạn mới (không thuộc phạm vi "inlier score" của V7).
- `generate_samples` (8.1% kernel V5) vẫn một thread vì phải giữ đúng dãy MT19937 và
  các lần loại mẫu trùng. Hướng tiếp (version riêng): sinh state MT19937 song song theo
  3 pha của bước twist (227/227/170 phần tử) trong shared memory, một thread chỉ tiêu
  thụ dãy số để chọn mẫu. Early termination/PROSAC đổi số vòng nên cũng phải là version
  riêng.
- Least-squares DLT của refinement giữ 1 thread để giữ thứ tự cộng của V0.
- Mỗi warp đọc 9 double của H từ global (broadcast); match được đọc coalesced và nằm
  gọn trong L2 (≤ 3.000 × 32 byte).

## 7. Trạng thái kiểm chứng

- Đã biên dịch host và device (sm_75) bằng clang 18 + header CUDA 12.4 cho
  `v7_stage5.cu`, `v7_pipeline.cu`, `main.cpp`, `test_v7_pipeline.cpp`; ptxas: kernel
  warp scoring 63 register, không spill.
- Đã mô phỏng CPU thuật toán warp (count, thứ tự cộng lỗi, argmax, compaction): trùng V1.
- **Chưa build bằng nvcc và chưa chạy trên GPU** (máy làm việc không có GPU). V7 chỉ
  được chốt khi `test_v7_pipeline` PASS và benchmark 12 case xong trên Colab.

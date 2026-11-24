# V3 pipeline — Crop với stack gộp trên nền V2

## 1. Phạm vi của V3

**V2** dùng lại S1–S3 và S5–S6 của V1. Ở S4, `v2_match` chia descriptor ứng viên thành
các tile trong shared memory để các thread trong cùng block dùng chung khi tính
khoảng cách Hamming. Cách lọc theo ratio test và kiểm tra hai chiều giữ theo V1.

**V3** kế thừa S1–S5 của V2, gồm cả S4 dùng descriptor tiling. Hai phiên bản chạy
chung luồng điều phối. V2 chọn `v1_warp_blend` và V3 chọn `v3_warp_blend` cho **S6**.
Hai bản gọi cùng `warp_blend_kernel`. V3 chỉ đổi kernel quét histogram để tìm
vùng crop. **V2 là baseline trực tiếp của V3.**

| Stage | Xử lý | Hàm gọi trong V3 |
|---|---|---|
| S1 | Chuyển sang ảnh xám và làm mờ Gaussian cho A và B | `v1_rgb_to_gray`, `v1_gaussian_blur` |
| S2 | Tính đáp ứng Harris, khử điểm không cực đại trong lân cận và lấy top-K | `v1_harris_response`, `v1_nms_topk` |
| S3 | Tạo descriptor BRIEF cho các điểm đặc trưng | `v1_describe` |
| S4 | Ghép descriptor theo khoảng cách Hamming với tiling trong shared memory, ratio test và kiểm tra hai chiều | `v2_match` |
| S5 | Ước lượng homography bằng RANSAC và tinh chỉnh mô hình | `v1_ransac_homography` |
| S6 | Tạo canvas, chiếu ngược, trộn màu và cắt panorama | `v3_warp_blend` |

Ảnh trái A là hệ tọa độ tham chiếu. Homography `H` ánh xạ ảnh phải B sang A.
[run_v3](../../src/cuda/v3_pipeline.cu) trả về thất bại khi đầu vào không phải
ảnh RGB hợp lệ, số cặp ghép ít hơn bốn, RANSAC không tìm được homography, S6
không tạo được panorama hợp lệ hoặc có lỗi CUDA. Luồng S1–S5 được định nghĩa ở
[v2_pipeline.cu](../../src/cuda/v2_pipeline.cu) và kernel matching ở
[v2_stage4.cu](../../src/stage_implementations/v2_stage4.cu).

## 2. Warp và trộn màu dùng chung với V2

1. **Tạo canvas.** `prepare_canvas_kernel` kiểm tra và chuẩn hóa `H`, tạo phép
   chiếu ngược, rồi chiếu bốn góc B sang hệ A. Khung bao của A và B sau phép chiếu
   cho gốc, chiều rộng và chiều cao của canvas. CPU nhận cấu hình này để cấp bộ
   nhớ và xác định kích thước lưới GPU.
2. **Đưa ảnh nguồn lên GPU.** Hai ảnh RGB được sao chép vào bộ đệm global memory.
   Kernel đọc mỗi kênh theo chỉ số `(y * width + x) * 3 + channel`.
3. **Chiếu ngược và nội suy.** Mỗi thread trong block `16 × 16` xử lý một pixel
   canvas. Thread kiểm tra tọa độ trong A và chiếu ngược sang B. Nếu tọa độ hợp
   lệ trong B, thread đọc bốn pixel lân cận rồi tính nội suy song tuyến tính
   bằng `double`.
4. **Trộn màu và ghi mask.** Nếu chỉ một ảnh phủ pixel, thread dùng màu của ảnh
   đó. Ở vùng chồng lấn, trọng số feather của mỗi ảnh tỷ lệ với khoảng cách đến
   biên gần nhất của ảnh tương ứng. Nếu cả hai khoảng cách bằng 0, mỗi ảnh có
   trọng số 0,5. Thread làm tròn màu về RGB 8 bit và đánh dấu pixel hợp lệ nếu
   ít nhất một ảnh phủ nó.

Canvas RGB và mask hợp lệ tiếp tục nằm trên GPU để dùng cho crop. Hàm chung
điều phối S6 nằm trong [v1_stage6.cu](../../src/stage_implementations/v1_stage6.cu).
V2 gọi kernel quét crop gốc. V3 gọi kernel quét crop gộp.

## 3. Crop bằng histogram và stack gộp

Mục tiêu là tìm **hình chữ nhật có diện tích lớn nhất mà mọi pixel đều hợp lệ**.
Với mỗi vị trí `(x, y)`, chiều cao histogram là số pixel hợp lệ liên tiếp ở cột
`x` tính ngược từ hàng `y` lên trên. Vì vậy, mỗi histogram giúp tìm các hình
chữ nhật có đáy nằm ở hàng tương ứng.

| Kernel | Cách chia việc |
|---|---|
| `build_crop_histograms_kernel` | Mỗi thread duyệt một cột từ trên xuống, tăng chiều cao khi gặp pixel hợp lệ và đặt lại về 0 khi gặp pixel không hợp lệ. Histogram được lưu theo hàng. |
| `find_crop_rows_compressed_stack_kernel` | Mỗi thread quét một hàng từ trái sang phải và chọn hình chữ nhật tốt nhất có đáy ở hàng đó. |
| `reduce_crop_rows_kernel` | Một block 256 thread chọn kết quả chung từ các hình chữ nhật tốt nhất của từng hàng. |
| `crop_image_kernel` | Mỗi thread chép ba kênh RGB của một pixel thuộc vùng đã chọn vào ảnh đầu ra. |

Trong bước quét hàng, mỗi chiều cao dương đang mở được lưu một lần dưới dạng
`(height, left)`, với `left` là cột xa nhất bên trái mà hình chữ nhật ở chiều
cao đó có thể bắt đầu. Ví dụ, dãy `[5, 5, 5, 5]` chỉ tạo một mục chiều cao 5.
Khi gặp cột thấp hơn ở vị trí `x`, thread lấy các mục cao hơn ra khỏi stack.
Mỗi mục tạo một ứng viên có chiều rộng `x - left`, chiều cao `height` và cạnh
trên ở hàng `y - height + 1`. Nếu chiều cao hiện tại cần một mục mới, nó có thể
kế thừa vị trí `left` từ các mục vừa lấy ra. Một cột ảo có chiều cao 0 ở cuối
hàng giúp xét hết các mục còn lại.

Mỗi mục dùng một từ 32 bit, gồm 16 bit cho chiều cao và 16 bit cho cột bắt đầu.
Ở V1, thread đẩy chỉ số của từng cột vào stack GPU, kể cả cột có chiều cao 0.
Để lấy chiều cao của mục trên cùng, thread phải tra histogram qua chỉ số
`heights[stack[stack_size - 1]]`. V3 lưu trực tiếp chiều cao và cột bắt đầu,
đồng thời giữ mục trên cùng trong biến cục bộ. Chẳng hạn, ở đoạn
`[5, 5, 5, 5]`, V1 tạo bốn mục và xét bốn ứng viên khi kết thúc đoạn.
V3 chỉ cần xét một ứng viên trải từ cột đầu đoạn, vì các ứng viên còn lại
có cùng chiều cao và chiều rộng nhỏ hơn. Bỏ các mục chiều cao 0 và gộp các
mục bằng nhau giúp giảm số lần truy cập stack và số ứng viên cần so sánh.
[Mã quét hàng V1](../../src/stage_implementations/v1_stage6.cu)

Việc duyệt histogram vẫn có độ phức tạp `O(width × height)`. GPU cấp đủ chỗ
cho trường hợp mỗi cột của một hàng có một chiều cao khác nhau.

Khi nhiều vùng có cùng diện tích, V3 ưu tiên vùng **rộng hơn**, rồi đến vùng có
**cạnh đáy ở hàng nhỏ hơn**, rồi **cạnh trái ở cột nhỏ hơn**, cùng thứ tự với V1.
CPU nhận tọa độ và kích thước crop để cấp bộ đệm đầu ra. GPU chép pixel và trả
ảnh RGB đã cắt về CPU. Toàn bộ S6 nằm trong
[v3_stage6.cu](../../src/stage_implementations/v3_stage6.cu).

## 4. Kiểm tra và benchmark

- [Test S4](../../src/tools/v2/test_matching.cpp) kiểm tra matching của V2 với V0/V1.
- [Test S6](../../src/tools/v3/test_v3_stage6.cu) so canvas, gốc canvas, mask,
  tọa độ crop và RGB với S6 của V1, cũng là S6 của V2. So với V0, sai số RGB tối đa được phép là 1
  mức màu và test ghi thêm MAE. Test bao gồm phép tịnh tiến, phép chiếu phối
  cảnh, kích thước biên, trường hợp đồng hạng và đầu vào lỗi.
- [Test toàn pipeline](../../src/tools/v3/test_v3_pipeline.cpp) chạy V0–V3,
  kiểm tra keypoint, match theo thứ tự, số inlier, sai số chiếu của homography
  tối đa `1e-3 px`, kích thước panorama và RGB. V2 và V3 phải có cùng hệ số
  homography và panorama trùng từng byte.
- [Notebook Colab](../../test_v3.ipynb) build một chương trình V0–V3 với
  `--fmad=false`, chạy kiểm tra và đo 12 cặp ảnh trên cùng GPU. Với mỗi phiên
  bản và mỗi cặp ảnh, CLI chạy một lượt làm nóng và ít nhất năm lượt đo.
  Notebook đảo thứ tự V2/V3 theo cảnh và ghi speedup so với V0 và phiên bản
  liền trước.

Bật GPU trong Colab và dùng checkout có bản crop gộp. Notebook mặc định lấy nhánh
`v3`. Với project tải lên Colab, dùng `UPDATE_REPO=False`. Đặt dataset trong
`dataset/` với bốn cảnh `location1`, `location2`, `location3` và `challenge`.
Mỗi cảnh có `small_720p`, `medium_1080p` và `large_4k` với hai tệp `1.jpg` và `2.jpg`.
Build dùng `PANORAMA_WITH_CUDA`, `PANORAMA_WITH_V2` và `PANORAMA_WITH_V3`.
Bản build liên kết mã V0/V1, `v2_pipeline.cu`, `v2_stage4.cu`, `v3_pipeline.cu`
và `v3_stage6.cu`. Các lệnh sau kiểm tra build và chạy V3 trên một cặp ảnh.

```bash
build/stitch_v3 --list-versions
build/stitch_v3 --selftest --version 3
mkdir -p out reports
build/stitch_v3 dataset/location1/small_720p/1.jpg \
  dataset/location1/small_720p/2.jpg out/v3.png \
  --version 3 --runs 5 --json reports/v3.json
```

### Cách đọc thời gian

- **S6 (`warp_blend`)** cộng thời gian các kernel tạo canvas, warp/blend và crop
  bằng CUDA events. `h2d` và `d2h` ghi riêng thời gian truyền dữ liệu của pipeline.
- **Host wall-clock (`end_to_end`)** là thời gian thực của cả pipeline S1–S6,
  đo bằng `std::chrono::steady_clock` trên CPU. Đồng hồ bắt đầu khi hai ảnh RGB
  đã ở trong RAM và dừng sau khi panorama đã về RAM và GPU đã đồng bộ. Số đo
  gồm cấp phát, gọi kernel, GPU xử lý, truyền dữ liệu và
  thời gian chờ. CLI đọc ảnh, chạy lượt làm nóng và ghi file ngoài các lượt đo.
  [Mã đo E2E](../../src/cuda/v2_pipeline.cu)

## 5. Profile V2 rồi kiểm tra V3

Notebook profile V2 trước khi chạy V3. Mặc định lấy `location1/medium_1080p`
và `location1/large_4k`. Báo thời gian warp, quét crop, tổng bốn kernel crop
và tỷ trọng crop trong kernel S6. Notebook `test_v3 (4).ipynb` chạy trên
Tesla T4 ở commit `6ead536`, mã lần đo `20261122T114605927358Z`.

| Ca profile V2 | Warp (ms) | Tổng crop (ms) | Crop trong S6 | Quét hàng trong crop |
|---|---:|---:|---:|---:|
| `location1/medium_1080p` | 17,466 | 12,900 | 42,5% | 95,3% |
| `location1/large_4k` | 54,770 | 33,754 | 38,1% | 90,7% |

Warp là phần tốn thời gian nhất trong S6 của V2 ở hai ca này. Crop vẫn chiếm
**38,1–42,5% thời gian kernel S6**, và quét hàng chiếm **hơn 90% thời gian crop**.
Vì vậy, V3 chọn quét hàng làm điểm tối ưu. Stack gộp giảm số mục phải ghi,
đọc và xét khi các cột có cùng chiều cao.

Sau bước baseline, notebook chạy test tọa độ crop và panorama, benchmark V0–V3
trên 12 cặp, rồi profile V3 trên cùng ca đã kiểm tra đầu ra. Bảng cuối tách
speedup kernel quét hàng, tổng crop, S6 và E2E. S6 và E2E lấy từ lượt benchmark
riêng. Hai trace phải có cùng `warp_blend_kernel` và `nearest_tiled_kernel`.

## 6. Kết quả V2 → V3 khi chỉ đổi crop

Cả **12/12 cặp ảnh PASS** kiểm tra đầu ra V0–V3, MAE bằng 0. Compute Sanitizer
báo **0 lỗi** ở bộ test S6. Bảng dùng trung bình nhân của các tỷ lệ thời gian.

| Độ phân giải | S6 V2/V3 | E2E V2/V3 | E2E V0/V3 | Ca S6 nhanh hơn V2 | Ca E2E nhanh hơn V2 |
|---|---:|---:|---:|---:|---:|
| 720p | 1,865× | 1,097× | 19,563× | 4/4 | 4/4 |
| 1080p | 1,658× | 1,092× | 20,616× | 4/4 | 2/4 |
| 4K | 1,355× | 1,034× | 17,479× | 4/4 | 3/4 |
| Cả 12 cặp | **1,612×** | **1,074×** | **19,174×** | **12/12** | **9/12** |

Thời gian S6 giảm khoảng **38,0%** và E2E giảm **6,9%** theo trung bình nhân.
Ba ca có E2E tăng là `location1/medium_1080p`, `challenge/medium_1080p` và
`challenge/large_4k`. Ví dụ ở `location1/medium_1080p`, S6 giảm từ **21,308
xuống 14,031 ms**, còn E2E gần như ngang nhau ở **103,289 và 103,605 ms**.
E2E còn tính truyền dữ liệu, cấp phát và các stage khác.

### Đối chiếu profiler trên cùng cặp ảnh

Hai bản đều chạy `warp_blend_kernel` ở S6 và `nearest_tiled_kernel` ở S4.
Nsight Systems ghi thời gian kernel trung bình trên các lượt trong trace.

| Ca profile | Quét crop V2 → V3 (ms) | Speedup quét crop | Tổng crop V2 → V3 (ms) | Speedup crop |
|---|---:|---:|---:|---:|
| `location1/medium_1080p` | 12,293 → 2,391 | **5,142×** | 12,900 → 2,990 | **4,314×** |
| `location1/large_4k` | 30,603 → 5,634 | **5,432×** | 33,754 → 8,847 | **3,815×** |

Thời gian quét hàng và tổng crop đều giảm trên hai ca được profile.
Phép so sánh này dùng cùng warp và chỉ đổi cách quét crop, nên cho thấy
tác động của stack gộp lên S6.

## 7. Kết quả lịch sử của bản thử nghiệm

Các số dưới đây thuộc bản V3 từng kết hợp **warp texture và crop gộp**.

Notebook kết quả `test_v3 (3).ipynb` chạy trên cùng Tesla T4 tại commit
`2a40679`. Benchmark và profiler thuộc lần đo `20261122T095700654885Z`.
Cả **12/12 cặp ảnh PASS**, MAE bằng 0 và Compute Sanitizer báo 0 lỗi cho bộ
test S6. Bảng lấy trung bình nhân của tỷ lệ thời gian V2 chia cho V3.

| Độ phân giải | Speedup S6 | Speedup toàn pipeline | S6 nhanh hơn | Toàn pipeline nhanh hơn |
|---|---:|---:|---:|---:|
| 720p | 1,85× | 1,13× | 4/4 | 4/4 |
| 1080p | 1,50× | 1,05× | 4/4 | 3/4 |
| 4K | 1,35× | 1,07× | 4/4 | 3/4 |
| Cả 12 cặp | **1,55×** | **1,08×** | **12/12** | **10/12** |

Tính theo thời gian, S6 giảm khoảng **35,6%** và toàn pipeline giảm khoảng
**7,6%**. Hai ca có E2E tăng là `location2/medium_1080p` và `location3/large_4k`.
Ở ca 4K này, S6 giảm từ **61,54 xuống 45,87 ms**, trong khi E2E tăng từ
**369,04 lên 385,14 ms** và D2H tăng từ **52,04 lên 57,21 ms**. Notebook chỉ
profile `location1/medium_1080p`, nên chưa xác định nguyên nhân của toàn bộ
chênh lệch E2E ở ca 4K.

### Profiler của `location1/medium_1080p`

Cả hai bản chạy `nearest_tiled_kernel` ở S4, xác nhận V3 kế thừa matching của
V2. Dưới đây là thời gian kernel trung bình lấy từ Nsight Systems.

| Phần xử lý | V2 (ms) | V3 (ms) |
|---|---:|---:|
| Warp/blend | 14,17 | 15,38 |
| Kernel quét hàng để tìm crop | 10,56 | 2,32 |
| Tổng bốn kernel crop | 11,15 | 2,92 |

Trong trace này, **kernel quét crop nhanh hơn 4,55×**. Phần thời gian giảm
ở crop lớn hơn phần tăng ở warp. Điều này phù hợp với cách stack gộp giảm số
mục phải ghi, đọc và xét khi các cột có cùng chiều cao. Speedup V2/V3 trong
bảng benchmark lịch sử tính cả hai thay đổi S6. Mục 6 ghi kết quả của phép
đo chỉ thay kernel quét crop.

### Lần đo trước khi ghép V2

Trong lần đo `20261122T080623810042Z` trên Tesla T4, bản thử nghiệm V3 còn dùng
S1–S5 của V1. So với V1, S6 nhanh hơn 1,58× và toàn pipeline nhanh hơn 1,12×
trên 12 cặp ảnh.

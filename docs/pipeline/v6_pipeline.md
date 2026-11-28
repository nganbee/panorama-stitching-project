# V6 — S4 một warp/query

Owner: Khánh. Source và notebook chuẩn bị với OpenAI Codex.

## Phạm vi

V6 thực hiện phân công S4 warp/query do Khánh chốt trong phiên làm việc này; mục V6 trong plan được cập nhật tương ứng. Bản triển khai này chỉ thử S4, không thay S6 và không xác nhận S4 là bottleneck trước khi đo V5.

V6 kế thừa V5: S1–S3, S5–S6, crop, thuật toán/tham số và device residency giữ nguyên. Chỉ lời gọi `v5::match` trong pipeline đổi thành `v6::match`.

## Implementation

128 thread/block = 4 warp = 4 query. Cả block nạp tile 128 descriptor B vào shared memory. Trong mỗi tile, lane l so candidate l, l+32, l+64, …; bốn query dùng chung tile. Mỗi lane giữ top-2 khóa 64 bit `(Hamming distance, candidate index)`; shuffle reduction 16/8/4/2/1 gộp top-2 của warp. Hai block barrier mỗi tile bảo đảm nạp xong trước khi đọc và đọc xong trước khi ghi tile kế tiếp. Warp không có query ở block cuối vẫn tham gia các barrier.

Kernel có shared-memory tile được khôi phục nguyên từ `64e6320`. So với V5, V6 đổi cách chia việc và tìm top-2 của S4, giữ tiling kế thừa V2. Bộ đo hiện tại vẫn dùng V5/V6 xen kẽ cùng process, một warm-up và năm mẫu mỗi bản. Cần chạy lại trên GPU để đánh giá, không dùng speedup cũ để kết luận kernel này nhanh hơn.

Index nhỏ nhất thắng khi bằng khoảng cách; hai candidate khác index có cùng distance vẫn chiếm cả best và second-best. Giữ sentinel 257 khi thiếu second neighbour. Ratio test strict, reverse cross-check và compaction theo idxA giữ nguyên V5. Warp không có query vẫn tham gia shuffle bằng sentinel. Hai chiều matching dùng cùng kernel.

## File và cách chạy

- `src/stage_implementations/v6_stage4.cu`: device matcher V6.
- `src/core_headers/v6_stages.h`: giao diện S4 device.
- `src/cuda/v6_pipeline.cu/.h`: V5 pipeline với matcher V6.
- `src/tools/v6/build.sh`: build CLI V0–V6 và các tool riêng V6.
- `src/tools/v6/test_matching.cu`: kiểm tra raw nearest/top-2 và ordered matches V0/V5/V6, gồm lane/stride boundaries, ties, ratio và cross-check. Test include implementation V6 trong cùng translation unit để kiểm tra kernel trước filter; build không link implementation đó lần thứ hai.
- `src/tools/v6/match_host.h`: adapter chỉ dùng trong test/verifier để cấp cùng descriptor đầu vào cho device matchers.
- `src/tools/v6/test_pipeline.cpp`: synthetic parity V0/V5/V6, lỗi đầu vào và residency bytes.
- `src/tools/v6/verify_pair.cpp`: kiểm chứng ảnh thật, JSON status/checks/metrics; dùng lại helper so sánh V2.
- `src/tools/v6/benchmark_pair.cpp`: 1 warm-up + 5 lượt đo pipeline, giữ raw samples và mean.
- `src/tools/v6/benchmark.py`: baseline V0/V5 giữ riêng; verification và đo lại V0/V5/V6 trên 12 case trong `comparison/`, V5/V6 đo xen kẽ trong cùng process. Bảng speedup chỉ dùng các số đo mới trong `comparison/`.
- `src/tools/v6/profile.py`: thu Nsight V5; `--case` chọn cùng input để profile V5/V6, report riêng theo case/version. Timing profiler không dùng tính speedup.
- `src/tools/v6/test_reporting.py`: test phát triển cho kiểm tra raw sample và lọc speedup theo correctness.
- `test_v6.ipynb`: 7 mục theo form V2; phần nhận xét để trống chờ output GPU.

Trên Colab GPU từ repo root:

```bash
bash src/tools/v6/build.sh
build/stage4_v6_selftest
build/test_v6_pipeline
build/stitch_v6 --selftest --version 6
```

Notebook tự chạy baseline, profiler và benchmark trong cùng session. Mỗi version dùng cùng ảnh/tham số/build. In GPU, báo S1–S6, kernel total, H2D/D2H, end-to-end; đọc/ghi ảnh ngoài timing. Lưu mean/min, raw samples, log và panorama; không vẽ speedup thành công cho case chưa qua verification.

## Tiêu chí

Same-input ordered matches phải trùng V0/V5. Pipeline V6 phải khớp chính xác V5 về matches, inlier, homography và panorama; so V0 với ngưỡng nhóm (projection ≤1e-3 px, cùng kích thước panorama, RGB lệch tối đa 1 mức). Hai bản cùng thất bại không được coi là panorama PASS.

Bảng timing/profiler V5 phải được đọc trước khi kết luận S4 đáng tối ưu. Nếu tỷ trọng S4 nhỏ, lợi ích end-to-end có thể nhỏ. Giữ cả case chậm, không viết nhận xét hoặc điền số đo trước khi chạy thực tế.

## Cách đo hiện tại

Baseline V0/V5 ở RUN_DIR dùng chọn tối ưu. Phase `optimized` tạo mới `RUN_DIR/comparison/`, kiểm chứng rồi đo lại V0 và cặp V5/V6. V0 dùng `std::chrono::high_resolution_clock`; mọi bản chạy trong cùng GPU session.

Cặp V5/V6 chạy bằng một lần gọi `benchmark_v6_pair --compare`: nạp ảnh một lần, cùng device/context, warm-up mỗi version một lần, sau đó đo năm vòng. Mỗi vòng gọi cả hai pipeline, đảo thứ tự ở vòng kế tiếp. Mỗi version có năm mẫu; một bản đứng trước ba lần, bản kia hai lần. Đổi bản bắt đầu ở case kế tiếp để giảm thiên lệch thứ tự. Không có image I/O giữa các lượt đo. Giữ event timing và đồng bộ của pipeline, báo mean/min và raw samples, không bỏ mẫu.

`<case>_paired.json` ghi mode, số warm-up, số mẫu và thứ tự; JSON V5/V6 ghi `sample_positions`. Python xác minh chúng khớp nhau trước khi tính speedup. Lỗi process/report làm cả cặp không hợp lệ; correctness không đạt không được tính speedup thành công. Không ghi đè baseline. Mỗi lần đo dùng RUN_DIR mới.

Profiler vẫn chạy riêng từng version trên cùng case, dùng phân tích kernel, không dùng tính speedup:

```bash
python src/tools/v6/profile.py --run-dir reports/v6_benchmark/RUN_ID/comparison --version 5 --case location1_large_4k
python src/tools/v6/profile.py --run-dir reports/v6_benchmark/RUN_ID/comparison --version 6 --case location1_large_4k
```

Output notebook được giữ nguyên, hiện có kết quả của kernel đọc trực tiếp global memory. Source đã khôi phục kernel shared-memory tile; cần Run all trên GPU để cập nhật kiểm chứng, timing, biểu đồ và profiling. Output đang lưu không xác nhận hiệu quả của kernel vừa khôi phục. Đo xen kẽ giúp cân bằng thứ tự, không bảo đảm loại hết dao động clock/nhiệt độ/tải hệ thống; vẫn phải xem raw samples và các stage không đổi khi nhận xét.

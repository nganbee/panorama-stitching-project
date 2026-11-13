## GPU PLAN

### Quy tắc version

V0 là bản C/C++ tuần tự, một thread, làm chuẩn cho correctness và speedup. **V1 phải chạy toàn bộ S1–S6 trên GPU bằng kernel cơ bản**; CPU chỉ đọc/ghi ảnh, điều phối launch và truyền dữ liệu, không tính thay bất kỳ stage nào. V2 trở đi kế thừa version trước và mỗi version có **một ý tối ưu chính** được chọn từ bottleneck đã đo. Giữ cùng dữ liệu đầu vào, thuật toán và tham số như V0 để so sánh công bằng; mọi version phải tạo kết quả tương đương V0.

Đề yêu cầu **tối thiểu bốn version: V0, V1 và ít nhất hai version tối ưu**. Mốc bàn giao ưu tiên của nhóm là V0–V3; V4–V7 là các hướng mở rộng theo thời gian và profiling, không phải số lượng bắt buộc của đề. Hai bản tối ưu đầu tiên vẫn phải có hai ý chính riêng biệt và đầy đủ kiểm chứng/bằng chứng đo; có thể thay ý dự kiến của V2/V3 khi bottleneck thực tế khác.

V1 phải mô tả cách ánh xạ công việc vào thread/block: ví dụ một thread/pixel, keypoint, query hoặc giả thuyết RANSAC. Các bước nhỏ hoặc cần giữ thứ tự có thể dùng một thread GPU; phần xử lý chính được chia song song theo cách cơ bản. Phạm vi GPU bao gồm cả sinh hệ số Gaussian, top-K, sinh mẫu/chọn mô hình/refinement S5, chuẩn bị ma trận/canvas và crop S6. CPU được kiểm tra metadata, cấp phát/giải phóng buffer, dùng count/kích thước do GPU trả về để launch/copy; không xử lý nội dung stage thay GPU.

S4 và S6 chiếm nhiều thời gian ở một số case V0, nên là ứng viên tối ưu sau V1. Thứ tự tối ưu V2–V7 phải được xác nhận lại bằng timing từng stage của **version liền trước**; nếu bottleneck đổi, nhóm đổi thứ tự và giải thích trong notebook.

### Các version dự kiến

| Version | Name | Requirement / ý tối ưu chính |
|---|---|---|
| **V0** | Sequential | S1–S6 chạy C/C++ một thread trên CPU; chuẩn correctness và speedup. |
| **V1** | Basic CUDA | **Cả S1–S6 chạy trên GPU** bằng kernel đơn giản, chủ yếu dùng global memory. CPU chỉ phụ trách I/O, launch và copy; có thể copy dữ liệu trung gian qua host để ghép stage, nhưng không dùng CPU fallback cho phép tính của stage nào. |
| **V2** | Optimized — S4 descriptor tiling | Nếu profiling V1 xác nhận S4 còn là bottleneck: tile descriptor B vào shared memory để các query tái sử dụng khi tính Hamming. Giữ V1 cho các stage khác. |
| **V3** | Optimized — S6 texture cache | Nếu profiling V2 xác nhận S6 còn là bottleneck: dùng texture cache để đọc pixel nguồn (point fetch), giữ bilinear thủ công, precision, feather và crop như V2. Ý chính chỉ là cách truy cập/cache ảnh nguồn. Kế thừa V2. |
| **V4** | Optimized — S1 blur tiling | Nếu profiling V3 xác nhận Gaussian blur còn tốn đáng kể: shared-memory tile có halo cho Gaussian blur, giữ kernel và quy tắc biên V0. Kế thừa V3. |
| **V5** | Optimized — GPU data residency | Nếu H2D/D2H giữa các stage chiếm tỷ lệ đáng kể: giữ ảnh, keypoints, descriptors, matches và homography trung gian trên device, bỏ copy các buffer tính toán qua CPU giữa stage; chỉ truyền input/output và metadata cần cho điều phối. Giữ thuật toán/kernel tính toán, kế thừa V4. |
| **V6** | Optimized — S4 warp/query | Nếu profiling V5 cho thấy S4 vẫn tốn thời gian: một warp cộng tác xử lý một query, dùng warp-shuffle reduction để tìm top-2 Hamming distance. Kế thừa V5 và giữ tiling của V2 nếu còn lợi. |
| **V7** | Optimized — S5 warp/hypothesis | Nếu profiling V6 cho thấy S5 trở thành bottleneck: một warp chấm inliers cho một giả thuyết RANSAC, giảm tổng inlier trong warp. Giữ nguyên mẫu, số vòng và ngưỡng của V0; kế thừa V6. |

Đây là thứ tự **dự kiến** dựa trên V0; trước khi chốt mỗi V2–V7 phải có timing của version trước chứng minh bottleneck liên quan. Nếu giả thuyết không đúng, thay ý tối ưu bằng bottleneck thực tế và ghi lý do. Version chậm hơn vẫn được giữ và phân tích; chỉ bỏ tối ưu trước đó khi notebook giải thích nguyên nhân. V2–V7 luôn phải qua kiểm tra kết quả so với V0.

#### Task breakdown theo version

**V0 — Sequential** Chốt 12 case, thuật toán/tham số, build và thiết bị; tổng hợp timing, keypoints, matches, inliers, homography và panorama CPU làm chuẩn. Các thành viên cung cấp kết quả stage mình phụ trách.

**V1 — Basic CUDA. Owner: Thịnh.** Chịu trách nhiệm ghép, kiểm tra và báo cáo bản chạy đủ S1–S6 trên GPU.

- **Scope của owner:** làm S1–S2 GPU (gray/blur/Harris/NMS/top-K); chốt giao diện giữa các stage, tích hợp `run_v1`, đo kernel, H2D/D2H và end-to-end, đối chiếu toàn pipeline với V0.
- **Đầu vào từ các thành viên:** Khánh bàn giao S3–S4 GPU (BRIEF/matching); Thụ bàn giao S5 GPU (sinh mẫu, RANSAC, chọn mô hình, inlier scoring/refinement); Ngân bàn giao S6 GPU (ma trận/canvas, inverse warp/bilinear/feather/crop) và hỗ trợ tích hợp. Mỗi người kiểm tra stage mình theo cùng tham số V0.
- **Mốc hoàn thành:** không còn phép tính S1–S6 trên CPU; keypoints, descriptors, matches, inliers, homography và panorama nhất quán với V0.

**V2 — S4 shared-memory descriptor tiling. Owner: Khánh.**

- **Scope của owner:** từ timing V1 xác nhận S4 còn là bottleneck; tile descriptor B vào shared memory để nhiều query cùng tái sử dụng, giữ ratio test/cross-check/tie-break; tích hợp với V1 và chạy benchmark S4 lẫn end-to-end.
- **Mốc hoàn thành:** V2 chỉ đổi S4 so với V1; matches và panorama nhất quán V0; ghi cả case chạy chậm và lý do.

**V3 — S6 texture cache. Owner: Ngân.**

- **Scope của owner:** từ timing và profiler của V2 xác nhận S6 còn là bottleneck; dùng texture point fetch cho ảnh nguồn, giữ bilinear thủ công, precision, feather và crop như V2/V0; tích hợp trên V2 và đo S6 kernel, copy, end-to-end. Hardware bilinear, đổi precision hoặc ROI là các thử nghiệm riêng, không gộp vào V3.
- **Mốc hoàn thành:** V3 chỉ đổi S6 so với V2; vùng hợp lệ, crop và panorama nhất quán V0.

**V4 — Gaussian blur shared-memory tile có halo. Owner: Thịnh.**

- **Scope của owner:** từ timing V3 xác nhận blur còn đáng kể; dùng shared-memory tile/halo trong S1, giữ hệ số Gaussian và xử lý biên; tích hợp trên V3 và đo blur lẫn end-to-end.
- **Mốc hoàn thành:** V4 chỉ đổi blur so với V3; blurred image, keypoints và panorama nhất quán V0.

**V5 — giữ dữ liệu trung gian trên GPU. Owner: Thụ.**

- **Scope của owner:** từ timing và profiler của V4 xác nhận copy giữa các stage đáng kể; phụ trách giao diện device của S5 (matches/keypoints vào, homography ra), phối hợp Thịnh và Khánh nối S1–S4, Ngân nối S6; tích hợp trên V4 để các buffer tính toán trung gian ở lại GPU. Chỉ copy input/output và metadata cần cho điều phối; giữ thuật toán/kernel tính toán để tách tác động của residency, đo H2D/D2H và end-to-end.
- **Mốc hoàn thành:** V5 chỉ đổi đường đi dữ liệu so với V4; kết quả nhất quán V0 và có số byte copy trước/sau.

**V6 — S4 một warp/query. Owner: Khánh.**

- **Scope của owner:** từ timing V5 xác nhận S4 còn tốn thời gian; cho một warp cùng so sánh descriptor B với một query rồi dùng shuffle reduction tìm best/second-best. Kế thừa shared-memory tiling V2 nếu còn lợi, giữ ratio test/cross-check/tie-break; đo riêng S4 và end-to-end.
- **Mốc hoàn thành:** V6 chỉ đổi cách chia việc/tìm top-2 của S4 so với V5; matches và panorama nhất quán V0.

**V7 — S5 một warp/RANSAC hypothesis. Owner: Thụ.**

- **Scope của owner:** từ timing V6 xác nhận S5 trở thành bottleneck; mỗi warp cùng chấm các match cho một hypothesis và reduction số inlier trong warp. Giữ nguyên dãy mẫu, 2.000 vòng, threshold và bước refinement; đo riêng S5 và end-to-end.
- **Mốc hoàn thành:** V7 chỉ đổi cách tính inlier score S5 so với V6; số inlier, homography và panorama nhất quán V0.

**Tổng hợp và Wrap up. Owner: Ngân**

- Tổng hợp các dữ liệu ở từng version để viết bảng report so sánh và báo cáo -> Rút ra nhận xét chung

### Thứ tự thực hiện

| Mốc / owner | Có thể làm song song từ khi | Cần trước khi chốt version | Output bàn giao |
|---|---|---|---|
| **V0 / cả nhóm** | Các stage CPU đã có thể kiểm tra riêng | Chốt 12 case, tham số và cách đo | Kết quả V0 từng stage, panorama chuẩn và timing baseline |
| **V1 / Thịnh** | Viết các stage lên GPU | Interface chung giữa S1→S6 và output chuẩn V0 | Bản full GPU chạy được, kiểm tra correctness và profile từng stage + copy |
| **V2 / Khánh** | Code song song V1 + S4 | V1 hoàn chỉnh và timing cho thấy S4 đáng tối ưu | Kernel S4 tiling + số đo V2 trên nền V1 |
| **V3 / Ngân** | Code song song V1 + S6 | V2 đã ghép và timing cho thấy S6 còn đáng tối ưu | Kernel S6 texture + số đo V3 trên nền V2 |
| **V4 / Thịnh** | Code song song V1 + Kernel gray/blur | V3 đã ghép và timing cho thấy blur còn đáng tối ưu | Kernel blur tile/halo + số đo V4 trên nền V3 |
| **V5 / Thụ** | V1 + Thiết kế đường đi dữ liệu song song | V4 đã ghép và timing cho thấy copy trung gian đáng kể | Pipeline giữ dữ liệu trên GPU + số byte copy và timing V5 |
| **V6 / Khánh** | Có S4 V1; có thể thử trên nhánh riêng trước khi V5 hoàn thành | V5 đã ghép, profiling S4; kết hợp được tối ưu V2 | Kernel warp/query + số đo V6 trên nền V5 |
| **V7 / Thụ** | Có S5 V1; có thể thử trên nhánh riêng trước khi V6 hoàn thành | V6 đã ghép và profiling cho thấy S5 đáng tối ưu | Kernel warp/hypothesis + số đo V7 trên nền V6 |

Ưu tiên **ghép, kiểm tra và báo cáo** V1 → V2 → V3 để đạt số version tối thiểu; nếu thực hiện phần mở rộng thì tiếp tục V4 → V5 → V6 → V7. Khi ghép một nhánh thử nghiệm vào version liền trước, owner chạy lại toàn bộ correctness và timing; version sau phải giữ tối ưu trước đó trừ khi notebook giải thích lý do bỏ. Nếu profiling không xác nhận bottleneck dự kiến, đổi ý tối ưu hoặc thứ tự version trước khi chốt; vẫn lưu kết quả thử nghiệm không hiệu quả để phân tích. Ngân tổng hợp bảng so sánh cuối từ số liệu mỗi owner bàn giao.


#### Cách đo và chọn tối ưu

- Dùng cùng input, GPU, CPU, build flags và tham số cho mọi version; chạy một warm-up rồi ít nhất năm lần đo. Thời gian end-to-end loại thời gian đọc/ghi ảnh như V0, nhưng **bao gồm** kernel launch, đồng bộ, cấp phát, H2D và D2H. Dùng CUDA events cho kernel/copy và đồng bộ trước khi dừng đồng hồ end-to-end. Ghi JSON theo từng case/version, không cộng kernel time vào end-to-end lần nữa.
- Chạy mọi version trong cùng một Colab GPU session, in tên GPU với từng kết quả; V0 dùng `std::chrono::high_resolution_clock`, GPU dùng CUDA events. Báo cáo trung bình ≥5 lần và thêm giá trị nhỏ nhất nếu phương sai lớn.
- Báo cáo `T_V0 / T_Vn` và `T_V(n-1) / T_Vn` cho **từng độ phân giải** và từng case, cùng bảng tổng hợp 12 case. Đo riêng từng stage/kernel, tổng kernel, H2D/D2H và end-to-end. Ghi cả version chậm hơn và phân tích nguyên nhân.
- Chỉ ước tính speedup của V2–V7 **sau khi đã đo V1**: dùng tỷ trọng thời gian của bottleneck ở version trước và mức tăng tốc giả định của đúng stage đó, rồi cộng chi phí copy/launch. Các ước tính 1.5×–3.2× của plan hybrid cũ không áp dụng vì V1 bây giờ phải chạy đủ S1–S6 trên GPU.
- Mỗi version tối ưu được thực hiện (V2 trở đi) dùng timing của version liền trước để chọn bottleneck và **phải có output profiler hỗ trợ lựa chọn đó**, như Nsight Systems/Compute. Lưu screenshot hoặc bản tóm tắt text lấy từ profiler, chỉ rõ version/case/GPU, stage/kernel chi phối và chi phí compute/copy liên quan; bảng timing đơn thuần không thay thế bằng chứng profiling. Kèm timing trước/sau và giải thích vì sao ý tối ưu có hoặc không có lợi.
- Với mọi version, kiểm tra đầu ra so với V0 theo tiêu chí ở phần dưới. Một chương trình CLI chọn được mọi version đã thực hiện; ngay khi thêm bất kỳ Vn từ V2 trở đi, mở rộng `versions.h` và CLI để chạy độc lập, không đợi đến V4.

#### Tiêu chí correctness giữa các version

Các ngưỡng dưới đây là **tiêu chí nghiệm thu ban đầu của nhóm**, không phải con số do đề quy định. Chốt bằng test V0/V1 trước khi đo các bản tối ưu, ghi trong notebook và dùng nhất quán cho mọi version/case. Nếu cần điều chỉnh phải có dữ liệu giải thích, cập nhật lại tiêu chí và kiểm tra lại các version liên quan; không nới ngưỡng riêng cho bản đang thất bại.

| Đầu ra / hợp đồng | Tiêu chí kiểm tra |
|---|---|
| Thuật toán và tham số | Giữ kernel Gaussian/quy tắc biên, BRIEF pattern, ratio/cross-check/tie-break, RANSAC seed/dãy mẫu, 2.000 vòng, threshold và refinement như V0 trong lộ trình V1–V7 hiện tại. |
| Grayscale / blurred / Harris | So từng pixel theo `abs_error <= atol + rtol * abs(V0)`. Ngưỡng ban đầu `(atol, rtol)`: grayscale `(1e-4, 1e-6)`, blur `(2e-4, 2e-6)`, Harris `(5e-2, 2e-5)`. Kiểm tra từng stage bằng cùng input để tách sai số lan truyền. |
| Keypoints | Trùng số lượng, tọa độ và thứ tự; score theo dung sai Harris. Test top-K vượt 3.000 điểm, response bằng nhau và tie-break raster. |
| Descriptors / matches | Khi nhận cùng input, descriptor trùng toàn bộ 256 bit; matches trùng số lượng, thứ tự và các trường `idxA`, `idxB`, `dist`. Đối chiếu lại chuỗi kết quả ở test end-to-end. |
| S5 | Cùng dãy mẫu và quy tắc chọn best; số inlier của H sau refinement phải trùng. So H bằng sai số chiếu tọa độ B→A trên các điểm kiểm tra và lưới trong ảnh B, tối đa `1e-3 px`, không chỉ so hệ số ma trận. |
| S6 | Kích thước canvas/crop và valid mask phải trùng; panorama cùng kích thước, sai số mỗi kênh RGB tối đa 1 mức trong thang 0–255, báo cáo thêm mean absolute error. |
| Ca lỗi | Kiểm tra input thiếu/không hợp lệ, dưới bốn matches hợp lệ, hình học suy biến và ca khó. Ghi status đúng như V0; CUDA error phải được báo, không fallback CPU. Hai bản cùng thất bại không được ghi là panorama correctness PASS. |

Sai số làm đổi tập keypoint/match/inlier hoặc crop phải được điều tra, không chỉ dựa vào panorama nhìn gần giống. Mỗi version cần test stage riêng và kiểm tra end-to-end trên đủ 12 case; ghi ca thất bại và nguyên nhân để phân tích.

#### Dữ liệu và notebook theo đề bài

- Dataset: ảnh tự chụp ở ≥3 địa điểm riêng, 3 cặp chuẩn + 1 cặp khó; overlap 30–50%, crop 16:9, mỗi cặp có 1280×720, 1920×1080 và 3840×2160. Chụp bằng cách đứng một chỗ và xoay máy để tránh parallax.
- Notebook phải chạy `Runtime → Run all` trên Colab GPU mới, lưu sẵn output cells, và mọi con số trong phần viết phải lấy từ output cell. Mỗi bản tối ưu trình bày đủ **bottleneck → idea → implementation → result → analysis**; bảng đóng góp ghi theo cả stage và version.
- Phần tổng hợp cần bảng/biểu đồ speedup theo version và độ phân giải, breakdown của version tốt nhất, gallery gồm ca lỗi, cùng README, slide, video 20–30 phút và khai báo nguồn/AI tools theo đề.

### Các phương pháp optimize cho từng stage

Các ý bên dưới là phương án dự phòng khi profiling chỉ ra bottleneck phù hợp; mỗi version optimized chỉ chọn **một** ý chính để đo tác động riêng. Các lựa chọn trong cùng một hàng là những thử nghiệm riêng, không mặc định gộp tất cả vào một version. Đổi precision hoặc dùng early termination phải được ghi thành thay đổi riêng, công khai tham số và kiểm tra lại hợp đồng correctness trước khi đưa vào lộ trình.

| Stage | Ý tưởng tối ưu nếu stage trở thành bottleneck | Rủi ro |
|---|---|---|
| S1 — RGB to grayscale | Mỗi CUDA thread xử lý một pixel; dùng vectorized load như `uchar4` để giảm số lần đọc global memory. Có thể fuse RGB-to-gray với horizontal Gaussian blur để giảm một lần ghi/đọc ảnh trung gian. Tái sử dụng device buffer giữa ảnh A và B. | Ảnh RGB interleaved cần được đọc đúng alignment. Phải giữ nguyên công thức grayscale và sai số float tương đương V0; fuse quá nhiều bước có thể làm tăng register pressure. |
| S1 — Gaussian blur | Giữ Gaussian blur separable gồm horizontal pass và vertical pass; đưa kernel cố định vào constant memory. Dùng shared-memory tiling với halo bán kính `r = 4` để tái sử dụng pixel lân cận và giảm global-memory traffic. | Phải giữ quy tắc clamp ở biên ảnh và kết quả gần tương đương V0. Tile hoặc halo quá lớn có thể làm giảm occupancy và tăng shared-memory usage. |
| S2 — Harris response | Dùng shared-memory tile với halo 1 pixel cho Sobel gradient; tính `Ix`, `Iy`, `Ixx`, `Iyy`, `Ixy` trong cùng kernel. Thay vòng lặp cửa sổ 5x5 độc lập bằng box filter separable hoặc rolling sum để giảm số lần đọc dữ liệu. | Phải giữ hệ số Sobel `1/8`, Harris `k = 0.04` và cách xử lý biên. Box filter hoặc fusion có thể tạo thêm buffer và làm tăng register pressure. |
| S2 — NMS và top-K | Mỗi thread kiểm tra một pixel ứng viên trong cửa sổ 5x5 bằng shared memory với halo 2; fuse threshold với NMS để loại sớm response nhỏ. Dùng prefix-sum hoặc `DeviceSelect` để compact candidate, sau đó dùng block-level top-K hoặc radix select trên GPU. | Phải giữ strict maximum, tie-break theo raster index, threshold 1% response lớn nhất, border 16 và thứ tự top-K deterministic. Sort toàn bộ candidate có thể tốn hơn local top-K khi số điểm nhỏ. |
| S3 | Mỗi CUDA thread xử lý một keypoint để tạo BRIEF 256 bit; thử constant memory cho 256 cặp offset cố định để tận dụng broadcast khi các thread đọc cùng offset. | Truy cập các offset khác nhau có thể bị tuần tự hóa; số keypoint ít có thể khiến chi phí truyền dữ liệu và gọi kernel lớn hơn lợi ích. Cần giữ nguyên pattern và thứ tự bit như V0. |
| S4 | Tính Hamming bằng XOR và `__popcll` trên bốn word 64 bit; tìm best và second-best trong một lượt. Thử riêng shared-memory tiling để tái sử dụng descriptor và một warp/query kết hợp shuffle reduction tìm top-2. | Tốn shared memory và chi phí đồng bộ; warp reduction có overhead. Phải xử lý đúng khoảng cách bằng nhau, giữ ratio test và cross-check hai chiều; cần đo riêng từng phương án để xác định hiệu quả. |
| S5 | Hướng đã chọn cho V7: một warp chấm inlier cho một hypothesis, giữ nguyên dãy mẫu, 2.000 vòng, threshold và refinement của V0. Early termination là hướng dự phòng cho một version/thử nghiệm riêng nếu profiling xác nhận phù hợp; không gộp vào V7 và không bật trong V1. | Đổi thứ tự cộng lỗi trong warp có thể ảnh hưởng tie-break; kiểm tra best model, inlier sau refinement, sai số chiếu và panorama. Early termination có thể bỏ qua mô hình tốt hơn nên cần công khai chính sách dừng, đánh giá riêng trên đủ 12 case và giải thích thay đổi so với baseline. Dùng timing GPU thực tế để chọn hướng, không suy từ tỷ trọng S5 CPU. |
| S6 | V3 chỉ thử texture cache bằng point fetch, giữ bilinear thủ công và precision. Hardware bilinear, đổi float/double hoặc ROI là các hướng riêng nếu profiling cho thấy phù hợp. Crop đã chạy GPU từ V1; nếu copy còn chi phối thì thử giữ canvas/output trung gian trên device và giảm copy trong V5, không tính việc chuyển crop từ CPU lên GPU là một tối ưu mới. | Texture cache có thể không nhanh hơn global memory; hardware interpolation hoặc đổi precision có thể gây sai khác và cần đánh giá riêng. Mọi hướng phải giữ đúng vùng hợp lệ, feather và crop; V5 đo cả số byte và thời gian copy trước/sau. |

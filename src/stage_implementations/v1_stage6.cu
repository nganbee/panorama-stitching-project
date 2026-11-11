// File: v1_stage6.cu
// Authors: Tran Kim Ngan
// Description: Ghép ảnh bằng inverse warp, nội suy song tuyến tính và feather blending.
//              CPU chuẩn bị phép biến đổi/canvas; GPU warp, tạo mask, tìm vùng crop và cắt ảnh.
// External sources: none
#include "../core_headers/v1_stages.h"
#include "../core_headers/stage6_crop.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <vector>

namespace {

constexpr int kMaxCanvasDimension = 32768;
constexpr size_t kMaxCanvasPixels = size_t{64} * 1024 * 1024;
constexpr double kRoundoffFactor = 64.0 * DBL_EPSILON;

#ifndef CUDA_CHECK
// Ghi mã lỗi CUDA và vị trí gọi ra stderr; macro này không dừng hay ném ngoại lệ.
#define CUDA_CHECK(call)                                                       \
    do {                                                                      \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__       \
                      << " code=" << err << " \"" << cudaGetErrorString(err)   \
                      << "\"\n";                                             \
        }                                                                     \
    } while (0)
#endif

// Vùng crop trên canvas: góc trên trái (x, y), chiều rộng w và chiều cao h.
using GpuRect = stage6_crop::Rect;

// =========================================================================
// 1. HOST PREPARATION HELPERS (O(1) Matrix & Bounding Box)
// =========================================================================

// CPU: chiếu điểm (x, y) qua homography h rồi chia cho tọa độ thuần nhất z.
// Ghi kết quả vào out_x/out_y; trả false nếu z quá gần 0 hoặc kết quả không hữu hạn.
// Dùng để chiếu bốn góc ảnh B sang hệ tọa độ ảnh A khi tính kích thước canvas.
bool project_host(const double h[9], double x, double y, double& out_x, double& out_y) {
    const double z = h[6] * x + h[7] * y + h[8];
    const double scale = std::abs(h[6] * x) + std::abs(h[7] * y) + std::abs(h[8]);
    if (!std::isfinite(z) || z == 0.0 || std::abs(z) <= kRoundoffFactor * scale) return false;
    out_x = (h[0] * x + h[1] * y + h[2]) / z;
    out_y = (h[3] * x + h[4] * y + h[5]) / z;
    return std::isfinite(out_x) && std::isfinite(out_y);
}

// CPU: đưa giá trị rất gần số nguyên về đúng số nguyên đó để khử sai số số thực.
// Tránh floor/ceil làm canvas rộng hoặc cao thêm một pixel ngoài dự kiến.
double snap_integer(double value) {
    const double nearest = std::round(value);
    return std::abs(value - nearest) <= 1e-8 + kRoundoffFactor * std::abs(value)
        ? nearest : value;
}

// CPU: kiểm tra H hữu hạn, chuẩn hóa theo hệ số có trị tuyệt đối lớn nhất,
// rồi tạo h (B -> A) và inverse (A -> B) cho phép chiếu ngược.
bool prepare_transform(const Homography& H, double h[9], double inverse[9]) {
    double scale = 0.0;
    for (double value : H.h) {
        if (!std::isfinite(value)) return false;
        scale = std::max(scale, std::abs(value));
    }
    if (scale == 0.0) return false;
    for (int i = 0; i < 9; ++i) h[i] = H.h[i] / scale;

    inverse[0] = h[4] * h[8] - h[5] * h[7];
    inverse[1] = h[2] * h[7] - h[1] * h[8];
    inverse[2] = h[1] * h[5] - h[2] * h[4];
    inverse[3] = h[5] * h[6] - h[3] * h[8];
    inverse[4] = h[0] * h[8] - h[2] * h[6];
    inverse[5] = h[2] * h[3] - h[0] * h[5];
    inverse[6] = h[3] * h[7] - h[4] * h[6];
    inverse[7] = h[1] * h[6] - h[0] * h[7];
    inverse[8] = h[0] * h[4] - h[1] * h[3];

    const double t0 = h[0] * inverse[0];
    const double t1 = h[1] * inverse[3];
    const double t2 = h[2] * inverse[6];
    const double determinant = t0 + t1 + t2;
    const double determinant_scale = std::abs(t0) + std::abs(t1) + std::abs(t2);
    if (!std::isfinite(determinant) || determinant == 0.0 ||
        std::abs(determinant) <= kRoundoffFactor * determinant_scale) return false;
    return true;
}

// =========================================================================
// 2. DEVICE KERNELS (WARP, FEATHER BLEND, HISTOGRAM SCAN, CROP)
// =========================================================================

// GPU: phiên bản chiếu điểm tương tự project_host, dùng trong kernel warp.
__device__ inline bool project_device(const double h[9], double x, double y, double& out_x, double& out_y) {
    const double z = h[6] * x + h[7] * y + h[8];
    const double scale = fabs(h[6] * x) + fabs(h[7] * y) + fabs(h[8]);
    if (!isfinite(z) || z == 0.0 || fabs(z) <= 64.0 * DBL_EPSILON * scale) return false;
    out_x = (h[0] * x + h[1] * y + h[2]) / z;
    out_y = (h[3] * x + h[4] * y + h[5]) / z;
    return isfinite(out_x) && isfinite(out_y);
}

// GPU: kẹp một kênh màu vào [0, 255], làm tròn tới số nguyên gần nhất
__device__ inline uint8_t rounded_channel_device(double value) {
    value = fmax(0.0, fmin(255.0, value));
    return static_cast<uint8_t>(floor(value + 0.5));
}

// KERNEL 1: mỗi thread tính một pixel trên canvas panorama.
__global__ void warp_blend_kernel(
    const uint8_t* __restrict__ A, int Aw, int Ah,
    const uint8_t* __restrict__ B, int Bw, int Bh,
    const Homography inv_H,
    int origin_x, int origin_y,
    int pano_w, int pano_h,
    uint8_t* __restrict__ pano,
    uint8_t* __restrict__ valid)
{
    const int px = blockIdx.x * blockDim.x + threadIdx.x;
    const int py = blockIdx.y * blockDim.y + threadIdx.y;

    if (px >= pano_w || py >= pano_h) return;

    const int ax = px + origin_x;
    const int ay = py + origin_y;
    const bool in_A = (ax >= 0 && ax < Aw && ay >= 0 && ay < Ah);

    double bx = 0.0, by = 0.0;
    const double sample_tolerance = 1e-7 + 64.0 * DBL_EPSILON * (Bw > Bh ? Bw : Bh);
    const bool in_B = project_device(inv_H.h, ax, ay, bx, by) &&
                      bx >= -sample_tolerance && bx <= (Bw - 1.0) + sample_tolerance &&
                      by >= -sample_tolerance && by <= (Bh - 1.0) + sample_tolerance;

    double color_B[3] = {0.0, 0.0, 0.0};
    if (in_B) {
        bx = fmax(0.0, fmin(static_cast<double>(Bw - 1), bx));
        by = fmax(0.0, fmin(static_cast<double>(Bh - 1), by));

        const int x0 = static_cast<int>(floor(bx));
        const int y0 = static_cast<int>(floor(by));
        const int x1 = (x0 + 1 < Bw - 1) ? x0 + 1 : (Bw - 1);
        const int y1 = (y0 + 1 < Bh - 1) ? y0 + 1 : (Bh - 1);

        const double dx = bx - x0;
        const double dy = by - y0;

        for (int c = 0; c < 3; ++c) {
            const double v00 = B[(static_cast<size_t>(y0) * Bw + x0) * 3 + c];
            const double v10 = B[(static_cast<size_t>(y0) * Bw + x1) * 3 + c];
            const double v01 = B[(static_cast<size_t>(y1) * Bw + x0) * 3 + c];
            const double v11 = B[(static_cast<size_t>(y1) * Bw + x1) * 3 + c];
            color_B[c] = (v00 * (1.0 - dx) + v10 * dx) * (1.0 - dy) +
                         (v01 * (1.0 - dx) + v11 * dx) * dy;
        }
    }

    const size_t pixel = static_cast<size_t>(py) * pano_w + px;
    const size_t source_A = in_A ? (static_cast<size_t>(ay) * Aw + ax) * 3 : 0;
    double weight_A = in_A ? 1.0 : 0.0;
    double weight_B = in_B ? 1.0 : 0.0;

    if (in_A && in_B) {
        const double dist_ax = (ax < Aw - 1 - ax) ? ax : (Aw - 1 - ax);
        const double dist_ay = (ay < Ah - 1 - ay) ? ay : (Ah - 1 - ay);
        const double distance_A = (dist_ax < dist_ay) ? dist_ax : dist_ay;

        const double dist_bx = (bx < Bw - 1.0 - bx) ? bx : (Bw - 1.0 - bx);
        const double dist_by = (by < Bh - 1.0 - by) ? by : (Bh - 1.0 - by);
        const double distance_B = (dist_bx < dist_by) ? dist_bx : dist_by;

        const double total = distance_A + distance_B;
        if (total > 0.0) {
            weight_A = distance_A / total;
            weight_B = distance_B / total;
        } else {
            weight_A = 0.5;
            weight_B = 0.5;
        }
    }

    for (int c = 0; c < 3; ++c) {
        const double color_A = in_A ? static_cast<double>(A[source_A + c]) : 0.0;
        pano[pixel * 3 + c] = rounded_channel_device(color_A * weight_A + color_B[c] * weight_B);
    }
    valid[pixel] = (in_A || in_B) ? 1 : 0;
}

// Crop phase 1: independent columns build all row histograms. Adjacent
// threads read/write adjacent columns; there is no cross-column dependency.
__global__ void build_crop_histograms_kernel(
    const uint8_t* __restrict__ valid, int width, int height,
    int* __restrict__ heights)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x >= width) return;
    int count = 0;
    for (int y = 0; y < height; ++y) {
        const size_t pixel = static_cast<size_t>(y) * width + x;
        count = valid[pixel] ? count + 1 : 0;
        heights[pixel] = count;
    }
}

// Crop phase 2: a separate thread scans each row, with its own stack.
// All rows run independently instead of one thread scanning the whole image.
__global__ void find_crop_rows_kernel(
    const int* __restrict__ heights, int width, int height,
    int* __restrict__ stacks, GpuRect* __restrict__ rows)
{
    const int y = blockIdx.x * blockDim.x + threadIdx.x;
    if (y >= height) return;
    const size_t offset = static_cast<size_t>(y) * width;
    rows[y] = stage6_crop::scan_row(heights + offset, width, y, stacks + offset);
}

// Crop phase 3: reduce the row winners with V0's deterministic tie order.
constexpr int kCropReduceThreads = 256;
__global__ void reduce_crop_rows_kernel(
    const GpuRect* __restrict__ rows, int height,
    GpuRect* __restrict__ out_rect)
{
    __shared__ GpuRect candidates[kCropReduceThreads];
    GpuRect best{};
    for (int y = threadIdx.x; y < height; y += blockDim.x) {
        if (stage6_crop::better(rows[y], best)) best = rows[y];
    }
    candidates[threadIdx.x] = best;
    __syncthreads();
    for (int stride = kCropReduceThreads / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride &&
            stage6_crop::better(candidates[threadIdx.x + stride], candidates[threadIdx.x])) {
            candidates[threadIdx.x] = candidates[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) *out_rect = candidates[0];
}

// KERNEL 3: mỗi thread sao chép ba kênh RGB của một pixel trong vùng crop.
__global__ void crop_image_kernel(
    const uint8_t* __restrict__ src, int src_w, int src_h,
    uint8_t* __restrict__ dst, int dst_w, int dst_h,
    int crop_x, int crop_y)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= dst_w || y >= dst_h) return;

    const size_t src_idx = (static_cast<size_t>(crop_y + y) * src_w + (crop_x + x)) * 3;
    const size_t dst_idx = (static_cast<size_t>(y) * dst_w + x) * 3;

    dst[dst_idx]     = src[src_idx];
    dst[dst_idx + 1] = src[src_idx + 1];
    dst[dst_idx + 2] = src[src_idx + 2];
}

}  // namespace

// =========================================================================
// 3. HÀM CHÍNH: v1_warp_blend (CPU ĐIỀU PHỐI, GPU XỬ LÝ ẢNH)
// =========================================================================

// Ghép hai ảnh RGB A, B thành pano; H ánh xạ tọa độ B sang tọa độ A.
// CPU kiểm tra đầu vào, chuẩn bị ma trận và canvas bao cả A lẫn B sau biến đổi.
// Sau khi chép ảnh lên GPU, lần lượt chạy warp/blend, tìm hình chữ nhật hợp lệ
// lớn nhất và crop. CPU đọc thông tin vùng crop để cấp phát ảnh kết quả,
// rồi nhận ảnh đã cắt về pano và giải phóng các buffer GPU.
// Đầu vào/phép biến đổi/canvas không hợp lệ hoặc không có vùng crop: pano rỗng.
// Lỗi CUDA được macro CUDA_CHECK ghi ra stderr, không tự hủy luồng xử lý.
void v1_warp_blend(const Image& A, const Image& B, const Homography& H, Image& pano) {
    // Đưa đầu ra về ảnh rỗng khi một bước kiểm tra không đạt.
    auto fail = [&]() { pano = Image{}; };
    if (!valid_rgb_image(A) || !valid_rgb_image(B)) { fail(); return; }

    double h[9], inverse[9];
    if (!prepare_transform(H, h, inverse)) { fail(); return; }

    // 1. Tính toán bounding box canvas trên Host (chỉ 4 góc, O(1))
    double min_x = 0.0, min_y = 0.0;
    double max_x = A.w - 1.0, max_y = A.h - 1.0;
    int denominator_sign = 0;
    for (int corner = 0; corner < 4; ++corner) {
        const double bx = (corner & 1) ? B.w - 1.0 : 0.0;
        const double by = (corner & 2) ? B.h - 1.0 : 0.0;
        double ax, ay;
        if (!project_host(h, bx, by, ax, ay)) { fail(); return; }
        const double z = h[6] * bx + h[7] * by + h[8];
        const int sign = z > 0.0 ? 1 : -1;
        if (denominator_sign != 0 && denominator_sign != sign) { fail(); return; }
        denominator_sign = sign;
        ax = snap_integer(ax);
        ay = snap_integer(ay);
        min_x = std::min(min_x, ax);
        min_y = std::min(min_y, ay);
        max_x = std::max(max_x, ax);
        max_y = std::max(max_y, ay);
    }
    min_x = std::floor(min_x);
    min_y = std::floor(min_y);
    max_x = std::ceil(max_x);
    max_y = std::ceil(max_y);

    const double width = max_x - min_x + 1.0;
    const double height = max_y - min_y + 1.0;
    if (!std::isfinite(width) || !std::isfinite(height) ||
        width <= 0.0 || height <= 0.0 ||
        width > kMaxCanvasDimension || height > kMaxCanvasDimension) {
        fail(); return;
    }

    const int origin_x = static_cast<int>(min_x), origin_y = static_cast<int>(min_y);
    const int pano_w = static_cast<int>(width), pano_h = static_cast<int>(height);
    const size_t pixels = static_cast<size_t>(pano_w) * pano_h;
    if (pixels > kMaxCanvasPixels) { fail(); return; }

    // 2. Cấp phát bộ nhớ Device (GPU)
    uint8_t *d_A = nullptr, *d_B = nullptr, *d_pano = nullptr, *d_valid = nullptr;
    int* d_crop_workspace = nullptr;

    const size_t bytes_A = static_cast<size_t>(A.w) * A.h * 3;
    const size_t bytes_B = static_cast<size_t>(B.w) * B.h * 3;
    const size_t bytes_pano = pixels * 3;
    const size_t bytes_valid = pixels;

    CUDA_CHECK(cudaMalloc(&d_A, bytes_A));
    CUDA_CHECK(cudaMalloc(&d_B, bytes_B));
    CUDA_CHECK(cudaMalloc(&d_pano, bytes_pano));
    CUDA_CHECK(cudaMalloc(&d_valid, bytes_valid));

    // Lưu trữ histogram
    const size_t workspace_bytes = 2 * pixels * sizeof(int) +
        (static_cast<size_t>(pano_h) + 1) * sizeof(GpuRect);
    CUDA_CHECK(cudaMalloc(&d_crop_workspace, workspace_bytes));
    int* d_heights = d_crop_workspace;
    int* d_stacks = d_crop_workspace + pixels;
    GpuRect* d_crop_rows = reinterpret_cast<GpuRect*>(d_crop_workspace + 2 * pixels);
    GpuRect* d_crop_rect = d_crop_rows + pano_h;

    // 3. Truyền dữ liệu Host -> Device (H2D)
    CUDA_CHECK(cudaMemcpy(d_A, A.data.data(), bytes_A, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, B.data.data(), bytes_B, cudaMemcpyHostToDevice));

    Homography inv_H;
    for (int i = 0; i < 9; ++i) inv_H.h[i] = inverse[i];

    // 4. KERNEL 1: Inverse Warping + Feather Blending (Toàn bộ trên GPU)
    dim3 block(16, 16);
    dim3 grid((pano_w + block.x - 1) / block.x, (pano_h + block.y - 1) / block.y);

    warp_blend_kernel<<<grid, block>>>(
        d_A, A.w, A.h,
        d_B, B.w, B.h,
        inv_H,
        origin_x, origin_y,
        pano_w, pano_h,
        d_pano, d_valid
    );
    CUDA_CHECK(cudaGetLastError());

    // 5. KERNEL 2: Quét biểu đồ histogram tìm Largest Valid Rectangle (Toàn bộ trên GPU)
    constexpr int crop_threads = 128;
    build_crop_histograms_kernel<<<(pano_w + crop_threads - 1) / crop_threads, crop_threads>>>(
        d_valid, pano_w, pano_h, d_heights);
    CUDA_CHECK(cudaGetLastError());
    find_crop_rows_kernel<<<(pano_h + crop_threads - 1) / crop_threads, crop_threads>>>(
        d_heights, pano_w, pano_h, d_stacks, d_crop_rows);
    CUDA_CHECK(cudaGetLastError());
    reduce_crop_rows_kernel<<<1, kCropReduceThreads>>>(d_crop_rows, pano_h, d_crop_rect);
    CUDA_CHECK(cudaGetLastError());

    // Đọc toạ độ crop về Host
    GpuRect crop;
    CUDA_CHECK(cudaMemcpy(&crop, d_crop_rect, sizeof(GpuRect), cudaMemcpyDeviceToHost));

    if (crop.w <= 0 || crop.h <= 0) {
        CUDA_CHECK(cudaFree(d_A));
        CUDA_CHECK(cudaFree(d_B));
        CUDA_CHECK(cudaFree(d_pano));
        CUDA_CHECK(cudaFree(d_valid));
        CUDA_CHECK(cudaFree(d_crop_workspace));
        fail();
        return;
    }

    // 6. KERNEL 3: Thực hiện Crop ảnh trực tiếp trên GPU
    uint8_t* d_cropped_pano = nullptr;
    const size_t bytes_cropped = static_cast<size_t>(crop.w) * crop.h * 3;
    CUDA_CHECK(cudaMalloc(&d_cropped_pano, bytes_cropped));

    dim3 crop_block(16, 16);
    dim3 crop_grid((crop.w + crop_block.x - 1) / crop_block.x,
                   (crop.h + crop_block.y - 1) / crop_block.y);

    crop_image_kernel<<<crop_grid, crop_block>>>(
        d_pano, pano_w, pano_h,
        d_cropped_pano, crop.w, crop.h,
        crop.x, crop.y
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // 7. Truyền kết quả Device -> Host (D2H)
    pano.w = crop.w;
    pano.h = crop.h;
    pano.c = 3;
    pano.data.resize(bytes_cropped);
    CUDA_CHECK(cudaMemcpy(pano.data.data(), d_cropped_pano, bytes_cropped, cudaMemcpyDeviceToHost));

    // 8. Giải phóng toàn bộ bộ nhớ GPU
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_pano));
    CUDA_CHECK(cudaFree(d_valid));
    CUDA_CHECK(cudaFree(d_crop_workspace));
    CUDA_CHECK(cudaFree(d_cropped_pano));
}

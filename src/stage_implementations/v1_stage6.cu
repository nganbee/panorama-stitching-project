// File: v1_stage6.cu
// Authors: Tran Kim Ngan
// Description: Pure CUDA inverse bilinear warp, feather blend, and GPU-side valid crop.
//              100% of calculations (warping, masking, histogram scan, cropping) run on GPU.
// External sources: none
#include "../core_headers/common.h"
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
#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__       \
                      << " code=" << err << " \"" << cudaGetErrorString(err)   \
                      << "\"\n";                                               \
        }                                                                      \
    } while (0)
#endif

struct GpuRect {
    int x = 0;
    int y = 0;
    int w = 0;
    int h = 0;
};

// =========================================================================
// 1. HOST PREPARATION HELPERS (O(1) Matrix & Bounding Box)
// =========================================================================

bool project_host(const double h[9], double x, double y, double& out_x, double& out_y) {
    const double z = h[6] * x + h[7] * y + h[8];
    const double scale = std::abs(h[6] * x) + std::abs(h[7] * y) + std::abs(h[8]);
    if (!std::isfinite(z) || z == 0.0 || std::abs(z) <= kRoundoffFactor * scale) return false;
    out_x = (h[0] * x + h[1] * y + h[2]) / z;
    out_y = (h[3] * x + h[4] * y + h[5]) / z;
    return std::isfinite(out_x) && std::isfinite(out_y);
}

double snap_integer(double value) {
    const double nearest = std::round(value);
    return std::abs(value - nearest) <= 1e-8 + kRoundoffFactor * std::abs(value)
        ? nearest : value;
}

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

__device__ inline bool project_device(const double h[9], double x, double y, double& out_x, double& out_y) {
    const double z = h[6] * x + h[7] * y + h[8];
    const double scale = fabs(h[6] * x) + fabs(h[7] * y) + fabs(h[8]);
    if (!isfinite(z) || z == 0.0 || fabs(z) <= 64.0 * DBL_EPSILON * scale) return false;
    out_x = (h[0] * x + h[1] * y + h[2]) / z;
    out_y = (h[3] * x + h[4] * y + h[5]) / z;
    return isfinite(out_x) && isfinite(out_y);
}

__device__ inline uint8_t rounded_channel_device(double value) {
    value = fmax(0.0, fmin(255.0, value));
    return static_cast<uint8_t>(floor(value + 0.5));
}

// KERNEL 1: Inverse Warping + Bilinear Sampling + Feather Blending
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

// KERNEL 2: Tìm hình chữ nhật hợp lệ lớn nhất trên GPU (Histogram Monotonic Stack)
__global__ void find_crop_rect_kernel(
    const uint8_t* __restrict__ valid,
    int width, int height,
    int* __restrict__ heights,
    int* __restrict__ stack,
    GpuRect* __restrict__ out_rect)
{
    // Chạy trên GPU: khởi tạo mảng heights ban đầu
    for (int x = threadIdx.x; x < width; x += blockDim.x) {
        heights[x] = 0;
    }
    __syncthreads();

    // Duyệt qua từng dòng của canvas
    GpuRect best = {0, 0, 0, 0};

    for (int y = 0; y < height; ++y) {
        const size_t row_offset = static_cast<size_t>(y) * width;
        // Cập nhật chiều cao song song bằng tất cả thread trong block
        for (int x = threadIdx.x; x < width; x += blockDim.x) {
            heights[x] = valid[row_offset + x] ? (heights[x] + 1) : 0;
        }
        __syncthreads();

        // Quét monotonic stack cho dòng y (thread 0 thực thi)
        if (threadIdx.x == 0) {
            int stack_size = 0;
            for (int x = 0; x <= width; ++x) {
                const int current = (x < width) ? heights[x] : 0;
                while (stack_size > 0 && current < heights[stack[stack_size - 1]]) {
                    const int bar = stack[--stack_size];
                    const int left = (stack_size == 0) ? 0 : (stack[stack_size - 1] + 1);
                    const int rect_w = x - left;
                    const int rect_h = heights[bar];
                    const long long area = static_cast<long long>(rect_w) * rect_h;
                    const long long best_area = static_cast<long long>(best.w) * best.h;
                    if (area > best_area || (area == best_area && rect_w > best.w)) {
                        best.x = left;
                        best.y = y - rect_h + 1;
                        best.w = rect_w;
                        best.h = rect_h;
                    }
                }
                if (x < width) {
                    stack[stack_size++] = x;
                }
            }
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        *out_rect = best;
    }
}

// KERNEL 3: Song song hoá việc cắt ảnh (Crop) ngay trên Device
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
// 3. HÀM INTERFACE CHÍNH: v1_warp_blend (100% PURE GPU PIPELINE)
// =========================================================================

void v1_warp_blend(const Image& A, const Image& B, const Homography& H, Image& pano) {
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
    int *d_heights = nullptr, *d_stack = nullptr;
    GpuRect *d_crop_rect = nullptr;

    const size_t bytes_A = static_cast<size_t>(A.w) * A.h * 3;
    const size_t bytes_B = static_cast<size_t>(B.w) * B.h * 3;
    const size_t bytes_pano = pixels * 3;
    const size_t bytes_valid = pixels;

    CUDA_CHECK(cudaMalloc(&d_A, bytes_A));
    CUDA_CHECK(cudaMalloc(&d_B, bytes_B));
    CUDA_CHECK(cudaMalloc(&d_pano, bytes_pano));
    CUDA_CHECK(cudaMalloc(&d_valid, bytes_valid));
    CUDA_CHECK(cudaMalloc(&d_heights, static_cast<size_t>(pano_w) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_stack, (static_cast<size_t>(pano_w) + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_crop_rect, sizeof(GpuRect)));

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
    find_crop_rect_kernel<<<1, 256>>>(
        d_valid,
        pano_w, pano_h,
        d_heights, d_stack,
        d_crop_rect
    );
    CUDA_CHECK(cudaGetLastError());

    // Đọc toạ độ crop (chỉ đúng 4 số nguyên int, siêu nhẹ) về Host
    GpuRect crop;
    CUDA_CHECK(cudaMemcpy(&crop, d_crop_rect, sizeof(GpuRect), cudaMemcpyDeviceToHost));

    if (crop.w <= 0 || crop.h <= 0) {
        CUDA_CHECK(cudaFree(d_A));
        CUDA_CHECK(cudaFree(d_B));
        CUDA_CHECK(cudaFree(d_pano));
        CUDA_CHECK(cudaFree(d_valid));
        CUDA_CHECK(cudaFree(d_heights));
        CUDA_CHECK(cudaFree(d_stack));
        CUDA_CHECK(cudaFree(d_crop_rect));
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

    // 7. Truyền kết quả Device -> Host (D2H): CHỈ truyền ảnh đã crop!
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
    CUDA_CHECK(cudaFree(d_heights));
    CUDA_CHECK(cudaFree(d_stack));
    CUDA_CHECK(cudaFree(d_crop_rect));
    CUDA_CHECK(cudaFree(d_cropped_pano));
}

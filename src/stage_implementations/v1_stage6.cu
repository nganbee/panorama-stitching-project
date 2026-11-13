// File: v1_stage6.cu
// V1 S6: all transform, canvas, warp, blending and crop computations on GPU.
#include "../core_headers/v1_stages.h"
#ifndef PANORAMA_STAGE6_HOST_TEST
#include "../core_headers/cuda_utils.cuh"
#endif
#include <cfloat>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <utility>


// V1-only crop helpers. Portable tests include this file in helper-test mode.
#ifdef __CUDACC__
#define STAGE6_HD __host__ __device__
#else
#define STAGE6_HD
#endif

namespace stage6_crop {
struct Rect {
    int x, y, w, h;
};

// V0 prefers area, width, then the first candidate in its row-major scan.
// An explicit total order preserves that choice when rows finish out of order.
STAGE6_HD inline bool better(const Rect& candidate, const Rect& best) {
    const long long area = static_cast<long long>(candidate.w) * candidate.h;
    const long long best_area = static_cast<long long>(best.w) * best.h;
    if (area != best_area) return area > best_area;
    if (candidate.w != best.w) return candidate.w > best.w;
    const int bottom = candidate.y + candidate.h;
    const int best_bottom = best.y + best.h;
    if (bottom != best_bottom) return bottom < best_bottom;
    return candidate.x < best.x;
}

// Each row owns its stack; no synchronization or atomics are needed here.
STAGE6_HD inline Rect scan_row(const int* heights, int width, int y, int* stack) {
    Rect best{};
    int stack_size = 0;
    for (int x = 0; x <= width; ++x) {
        const int current = x < width ? heights[x] : 0;
        while (stack_size > 0 && current < heights[stack[stack_size - 1]]) {
            const int bar = stack[--stack_size];
            const int left = stack_size == 0 ? 0 : stack[stack_size - 1] + 1;
            const Rect candidate{left, y - heights[bar] + 1, x - left, heights[bar]};
            if (better(candidate, best)) best = candidate;
        }
        if (x < width) stack[stack_size++] = x;
    }
    return best;
}
} // namespace stage6_crop

#undef STAGE6_HD

#ifdef __CUDACC__
#define S6_HD __host__ __device__
#else
#define S6_HD
#endif
namespace {
constexpr int kMaxCanvasDimension = 32768;
constexpr size_t kMaxCanvasPixels = size_t{64} * 1024 * 1024;
constexpr double kRoundoffFactor = 64.0 * DBL_EPSILON;
using GpuRect = stage6_crop::Rect;
S6_HD bool finite(double x) { return x >= -DBL_MAX && x <= DBL_MAX; }
S6_HD bool project_point(const double h[9], double x, double y, double& out_x, double& out_y) {
    const double z = h[6] * x + h[7] * y + h[8];
    const double scale = fabs(h[6] * x) + fabs(h[7] * y) + fabs(h[8]);
    if (!finite(z) || z == 0.0 || fabs(z) <= kRoundoffFactor * scale) return false;
    out_x = (h[0] * x + h[1] * y + h[2]) / z;
    out_y = (h[3] * x + h[4] * y + h[5]) / z;
    return finite(out_x) && finite(out_y);
}

// GPU helper: đưa giá trị rất gần số nguyên về đúng số nguyên đó để khử sai số số thực.
// Tránh floor/ceil làm canvas rộng hoặc cao thêm một pixel ngoài dự kiến.
S6_HD double snap_integer(double value) {
    const double nearest = round(value);
    return fabs(value - nearest) <= 1e-8 + kRoundoffFactor * fabs(value)
        ? nearest : value;
}

// GPU helper: kiểm tra H hữu hạn, chuẩn hóa theo hệ số có trị tuyệt đối lớn nhất,
// rồi tạo h (B -> A) và inverse (A -> B) cho phép chiếu ngược.
S6_HD bool prepare_transform(const Homography& H, double h[9], double inverse[9]) {
    double scale = 0.0;
    for (double value : H.h) {
        if (!finite(value)) return false;
        scale = fmax(scale, fabs(value));
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
    const double determinant_scale = fabs(t0) + fabs(t1) + fabs(t2);
    if (!finite(determinant) || determinant == 0.0 ||
        fabs(determinant) <= kRoundoffFactor * determinant_scale) return false;
    return true;
}


struct CanvasConfig {
    Homography inverse{};
    int origin_x = 0, origin_y = 0, width = 0, height = 0, ok = 0;
};

S6_HD CanvasConfig make_canvas(int aw, int ah, int bw, int bh, const Homography& H) {
    CanvasConfig result{};
    double h[9];
    if (!prepare_transform(H, h, result.inverse.h)) return result;
    double min_x = 0, min_y = 0, max_x = aw - 1.0, max_y = ah - 1.0;
    int denominator_sign = 0;
    for (int corner = 0; corner < 4; ++corner) {
        const double bx = (corner & 1) ? bw - 1.0 : 0.0;
        const double by = (corner & 2) ? bh - 1.0 : 0.0;
        double ax, ay;
        if (!project_point(h, bx, by, ax, ay)) return result;
        const double z = h[6] * bx + h[7] * by + h[8];
        const int sign = z > 0 ? 1 : -1;
        if (denominator_sign != 0 && denominator_sign != sign) return result;
        denominator_sign = sign;
        ax = snap_integer(ax); ay = snap_integer(ay);
        min_x = fmin(min_x, ax); min_y = fmin(min_y, ay);
        max_x = fmax(max_x, ax); max_y = fmax(max_y, ay);
    }
    min_x = floor(min_x); min_y = floor(min_y);
    max_x = ceil(max_x); max_y = ceil(max_y);
    const double width = max_x - min_x + 1.0, height = max_y - min_y + 1.0;
    if (!finite(width) || !finite(height) || width <= 0 || height <= 0 ||
        width > kMaxCanvasDimension || height > kMaxCanvasDimension) return result;
    result.width = static_cast<int>(width); result.height = static_cast<int>(height);
    if (static_cast<size_t>(result.width) * result.height > kMaxCanvasPixels) return result;
    result.origin_x = static_cast<int>(min_x); result.origin_y = static_cast<int>(min_y);
    result.ok = 1;
    return result;
}

#ifndef PANORAMA_STAGE6_HOST_TEST
__global__ void prepare_canvas_kernel(int aw, int ah, int bw, int bh,
                                       Homography H, CanvasConfig* config) {
    if (blockIdx.x || threadIdx.x) return;
    *config = make_canvas(aw, ah, bw, bh, H);
}

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
    const bool in_B = project_point(inv_H.h, ax, ay, bx, by) &&
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


#endif // PANORAMA_STAGE6_HOST_TEST
} // namespace
#undef S6_HD

#ifndef PANORAMA_STAGE6_HOST_TEST
void v1_warp_blend(const Image& A, const Image& B, const Homography& H, Image& pano) {
    pano = Image{};
    if (!valid_rgb_image(A) || !valid_rgb_image(B)) return;
    v1_cuda::Buffer<CanvasConfig> device_config(1);
    v1_cuda::launch([&] { prepare_canvas_kernel<<<1, 1>>>(
        A.w, A.h, B.w, B.h, H, device_config.data); });
    CanvasConfig config;
    v1_cuda::copy(&config, device_config.data, sizeof(config), cudaMemcpyDeviceToHost);
    if (!config.ok) return;

    // CPU uses GPU-produced dimensions only for allocation and launch geometry.
    const int width = config.width, height = config.height;
    const size_t pixels = static_cast<size_t>(width) * height;
    v1_cuda::Buffer<uint8_t> a(A.data.size()), b(B.data.size()), canvas(pixels * 3), valid(pixels);
    v1_cuda::Buffer<int> heights(pixels), stacks(pixels);
    v1_cuda::Buffer<GpuRect> rows(height), rect(1);
    v1_cuda::copy(a.data, A.data.data(), A.data.size(), cudaMemcpyHostToDevice);
    v1_cuda::copy(b.data, B.data.data(), B.data.size(), cudaMemcpyHostToDevice);
    const dim3 block(16, 16);
    const dim3 grid((width + 15) / 16, (height + 15) / 16);
    v1_cuda::launch([&] { warp_blend_kernel<<<grid, block>>>(
        a.data, A.w, A.h, b.data, B.w, B.h, config.inverse,
        config.origin_x, config.origin_y, width, height, canvas.data, valid.data); });
    constexpr int threads = 128;
    v1_cuda::launch([&] { build_crop_histograms_kernel<<<(width + threads - 1) / threads, threads>>>(
        valid.data, width, height, heights.data); });
    v1_cuda::launch([&] { find_crop_rows_kernel<<<(height + threads - 1) / threads, threads>>>(
        heights.data, width, height, stacks.data, rows.data); });
    v1_cuda::launch([&] { reduce_crop_rows_kernel<<<1, kCropReduceThreads>>>(rows.data, height, rect.data); });
    GpuRect crop{};
    v1_cuda::copy(&crop, rect.data, sizeof(crop), cudaMemcpyDeviceToHost);
    if (crop.w <= 0 || crop.h <= 0) return;

    const size_t bytes = static_cast<size_t>(crop.w) * crop.h * 3;
    v1_cuda::Buffer<uint8_t> cropped(bytes);
    const dim3 crop_grid((crop.w + 15) / 16, (crop.h + 15) / 16);
    v1_cuda::launch([&] { crop_image_kernel<<<crop_grid, block>>>(
        canvas.data, width, height, cropped.data, crop.w, crop.h, crop.x, crop.y); });
    Image result;
    result.w = crop.w; result.h = crop.h; result.c = 3; result.data.resize(bytes);
    v1_cuda::copy(result.data.data(), cropped.data, bytes, cudaMemcpyDeviceToHost);
    pano = std::move(result);
}
#endif // PANORAMA_STAGE6_HOST_TEST

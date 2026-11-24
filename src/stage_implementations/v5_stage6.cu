// File: v5_stage6.cu
// Authors: Team 8 (Thụ: V5 owner)
// Description: V5 device-resident S6 warp/blend/crop, inheriting V4 warp/blend + V3 compressed crop stack.
#include "../core_headers/v5_stages.h"
#include "../core_headers/v3_stages.h"
#include "../core_headers/cuda_utils.cuh"
#include <cfloat>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <utility>

#ifndef PANORAMA_V4_WARP_BLOCK_X
#define PANORAMA_V4_WARP_BLOCK_X 32
#endif
#ifndef PANORAMA_V4_WARP_BLOCK_Y
#define PANORAMA_V4_WARP_BLOCK_Y 8
#endif

namespace {
using stage6_detail::kMaxCanvasDimension;
constexpr size_t kMaxCanvasPixels = size_t{64} * 1024 * 1024;
constexpr double kRoundoffFactor = 64.0 * DBL_EPSILON;
using GpuRect = stage6_crop::Rect;

__device__ bool finite_val(double x) { return x >= -DBL_MAX && x <= DBL_MAX; }

__device__ bool project_point_inv(const double h[9], double x, double y,
                                  double& out_x, double& out_y) {
    const double z = h[6] * x + h[7] * y + h[8];
    const double scale = fabs(h[6] * x) + fabs(h[7] * y) + fabs(h[8]);
    if (!finite_val(z) || z == 0.0 || fabs(z) <= kRoundoffFactor * scale) return false;
    const double inv_z = 1.0 / z;
    out_x = (h[0] * x + h[1] * y + h[2]) * inv_z;
    out_y = (h[3] * x + h[4] * y + h[5]) * inv_z;
    return finite_val(out_x) && finite_val(out_y);
}

__device__ double snap_integer(double value) {
    const double nearest = round(value);
    return fabs(value - nearest) <= 1e-8 + kRoundoffFactor * fabs(value)
        ? nearest : value;
}

__device__ bool prepare_transform(const Homography& H, double h[9], double inverse[9]) {
    double scale = 0.0;
    for (double value : H.h) {
        if (!finite_val(value)) return false;
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
    if (!finite_val(determinant) || determinant == 0.0 ||
        fabs(determinant) <= kRoundoffFactor * determinant_scale) return false;
    return true;
}

struct CanvasConfig {
    Homography inverse{};
    int origin_x = 0, origin_y = 0, width = 0, height = 0, ok = 0;
};

__device__ CanvasConfig make_canvas(int aw, int ah, int bw, int bh, const Homography& H) {
    CanvasConfig result{};
    double h[9];
    if (!prepare_transform(H, h, result.inverse.h)) return result;
    double min_x = 0, min_y = 0, max_x = aw - 1.0, max_y = ah - 1.0;
    int denominator_sign = 0;
    for (int corner = 0; corner < 4; ++corner) {
        const double bx = (corner & 1) ? bw - 1.0 : 0.0;
        const double by = (corner & 2) ? bh - 1.0 : 0.0;
        double ax, ay;
        if (!project_point_inv(h, bx, by, ax, ay)) return result;
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
    if (!finite_val(width) || !finite_val(height) || width <= 0 || height <= 0 ||
        width > kMaxCanvasDimension || height > kMaxCanvasDimension) return result;
    result.width = static_cast<int>(width); result.height = static_cast<int>(height);
    if (static_cast<size_t>(result.width) * result.height > kMaxCanvasPixels) return result;
    result.origin_x = static_cast<int>(min_x); result.origin_y = static_cast<int>(min_y);
    result.ok = 1;
    return result;
}

__global__ void prepare_canvas_from_s5_kernel(int aw, int ah, int bw, int bh,
                                              const V5HomographyResult* s5,
                                              CanvasConfig* config) {
    if (blockIdx.x || threadIdx.x) return;
    if (!s5 || !s5->ok) {
        *config = CanvasConfig{};
        return;
    }
    *config = make_canvas(aw, ah, bw, bh, s5->H);
}

__global__ void pack_rgb_to_uchar4_kernel(const uint8_t* __restrict__ rgb,
                                          size_t pixels,
                                          uchar4* __restrict__ rgba) {
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= pixels) return;
    const size_t base = idx * 3;
    rgba[idx] = make_uchar4(__ldg(&rgb[base]), __ldg(&rgb[base + 1]), __ldg(&rgb[base + 2]), 255);
}

__device__ inline uint8_t rounded_channel_f32(float value) {
    value = fmaxf(0.0f, fminf(255.0f, value));
    return static_cast<uint8_t>(floorf(value + 0.5f));
}

__global__ void warp_blend_v5_kernel(
    const uint8_t* __restrict__ A, int Aw, int Ah,
    const uchar4* __restrict__ B_rgba, int Bw, int Bh,
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
    const bool in_B = project_point_inv(inv_H.h, ax, ay, bx, by) &&
                      bx >= -sample_tolerance && bx <= (Bw - 1.0) + sample_tolerance &&
                      by >= -sample_tolerance && by <= (Bh - 1.0) + sample_tolerance;

    const size_t pixel = static_cast<size_t>(py) * pano_w + px;
    const size_t out_idx = pixel * 3;

    if (!in_A && !in_B) {
        pano[out_idx] = 0;
        pano[out_idx + 1] = 0;
        pano[out_idx + 2] = 0;
        valid[pixel] = 0;
        return;
    }

    if (in_A && !in_B) {
        const size_t src_A = (static_cast<size_t>(ay) * Aw + ax) * 3;
        pano[out_idx]     = __ldg(&A[src_A]);
        pano[out_idx + 1] = __ldg(&A[src_A + 1]);
        pano[out_idx + 2] = __ldg(&A[src_A + 2]);
        valid[pixel] = 1;
        return;
    }

    bx = fmax(0.0, fmin(static_cast<double>(Bw - 1), bx));
    by = fmax(0.0, fmin(static_cast<double>(Bh - 1), by));

    const int x0 = static_cast<int>(floor(bx));
    const int y0 = static_cast<int>(floor(by));
    const int x1 = (x0 + 1 < Bw - 1) ? x0 + 1 : (Bw - 1);
    const int y1 = (y0 + 1 < Bh - 1) ? y0 + 1 : (Bh - 1);

    const float dx = static_cast<float>(bx - x0);
    const float dy = static_cast<float>(by - y0);
    const float w00 = (1.0f - dx) * (1.0f - dy);
    const float w10 = dx * (1.0f - dy);
    const float w01 = (1.0f - dx) * dy;
    const float w11 = dx * dy;

    const size_t row0 = static_cast<size_t>(y0) * Bw;
    const size_t row1 = static_cast<size_t>(y1) * Bw;
    const uchar4 p00 = __ldg(&B_rgba[row0 + x0]);
    const uchar4 p10 = __ldg(&B_rgba[row0 + x1]);
    const uchar4 p01 = __ldg(&B_rgba[row1 + x0]);
    const uchar4 p11 = __ldg(&B_rgba[row1 + x1]);

    const float b_r = p00.x * w00 + p10.x * w10 + p01.x * w01 + p11.x * w11;
    const float b_g = p00.y * w00 + p10.y * w10 + p01.y * w01 + p11.y * w11;
    const float b_b = p00.z * w00 + p10.z * w10 + p01.z * w01 + p11.z * w11;

    if (!in_A) {
        pano[out_idx]     = rounded_channel_f32(b_r);
        pano[out_idx + 1] = rounded_channel_f32(b_g);
        pano[out_idx + 2] = rounded_channel_f32(b_b);
        valid[pixel] = 1;
        return;
    }

    const float dist_ax = static_cast<float>((ax < Aw - 1 - ax) ? ax : (Aw - 1 - ax));
    const float dist_ay = static_cast<float>((ay < Ah - 1 - ay) ? ay : (Ah - 1 - ay));
    const float distance_A = fminf(dist_ax, dist_ay);

    const float bx_f = static_cast<float>(bx);
    const float by_f = static_cast<float>(by);
    const float dist_bx = fminf(bx_f, static_cast<float>(Bw - 1) - bx_f);
    const float dist_by = fminf(by_f, static_cast<float>(Bh - 1) - by_f);
    const float distance_B = fminf(dist_bx, dist_by);

    const float total = distance_A + distance_B;
    float weight_A = 0.5f;
    float weight_B = 0.5f;
    if (total > 0.0f) {
        weight_A = distance_A / total;
        weight_B = distance_B / total;
    }

    const size_t src_A = (static_cast<size_t>(ay) * Aw + ax) * 3;
    const float a_r = static_cast<float>(__ldg(&A[src_A]));
    const float a_g = static_cast<float>(__ldg(&A[src_A + 1]));
    const float a_b = static_cast<float>(__ldg(&A[src_A + 2]));

    pano[out_idx]     = rounded_channel_f32(a_r * weight_A + b_r * weight_B);
    pano[out_idx + 1] = rounded_channel_f32(a_g * weight_A + b_g * weight_B);
    pano[out_idx + 2] = rounded_channel_f32(a_b * weight_A + b_b * weight_B);
    valid[pixel] = 1;
}

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

    dst[dst_idx]     = __ldg(&src[src_idx]);
    dst[dst_idx + 1] = __ldg(&src[src_idx + 1]);
    dst[dst_idx + 2] = __ldg(&src[src_idx + 2]);
}

constexpr int kCropThreads = 128;

GpuRect find_crop_v5(const uint8_t* valid, int width, int height, int* heights, int* stacks,
                     GpuRect* rows, GpuRect* rect) {
    v1_cuda::launch([&] { build_crop_histograms_kernel<<<(width + kCropThreads - 1) / kCropThreads, kCropThreads>>>(
        valid, width, height, heights); });
    stage6_detail::launch_compressed_crop_rows(heights, width, height, stacks, rows);
    v1_cuda::launch([&] { reduce_crop_rows_kernel<<<1, kCropReduceThreads>>>(rows, height, rect); });
    GpuRect crop{};
    v1_cuda::copy(&crop, rect, sizeof(crop), cudaMemcpyDeviceToHost);
    return crop;
}
} // namespace

namespace v5 {
void warp_blend(const uint8_t* A, int Aw, int Ah, const uint8_t* B, int Bw, int Bh,
                const V5HomographyResult* device_result, Image& pano) {
    pano = Image{};
    if (!A || !B || !device_result || Aw <= 0 || Ah <= 0 || Bw <= 0 || Bh <= 0) return;
    v1_cuda::Buffer<CanvasConfig> device_config(1);
    v1_cuda::launch([&] { prepare_canvas_from_s5_kernel<<<1, 1>>>(
        Aw, Ah, Bw, Bh, device_result, device_config.data); });
    CanvasConfig config;
    v1_cuda::copy(&config, device_config.data, sizeof(config), cudaMemcpyDeviceToHost);
    if (!config.ok) return;

    const int width = config.width, height = config.height;
    const size_t pixels = static_cast<size_t>(width) * height;
    const size_t b_pixels = static_cast<size_t>(Bw) * Bh;
    v1_cuda::Buffer<uint8_t> canvas(pixels * 3), valid(pixels);
    v1_cuda::Buffer<uchar4> b_rgba(b_pixels);
    v1_cuda::Buffer<int> heights(pixels), stacks(pixels);
    v1_cuda::Buffer<GpuRect> rows(height), rect(1);

    constexpr int kPackThreads = 256;
    v1_cuda::launch([&] { pack_rgb_to_uchar4_kernel<<<(b_pixels + kPackThreads - 1) / kPackThreads, kPackThreads>>>(
        B, b_pixels, b_rgba.data); });

    const dim3 warp_block(PANORAMA_V4_WARP_BLOCK_X, PANORAMA_V4_WARP_BLOCK_Y);
    const dim3 warp_grid((width + warp_block.x - 1) / warp_block.x,
                         (height + warp_block.y - 1) / warp_block.y);
    v1_cuda::launch([&] { warp_blend_v5_kernel<<<warp_grid, warp_block>>>(
        A, Aw, Ah, b_rgba.data, Bw, Bh, config.inverse,
        config.origin_x, config.origin_y, width, height, canvas.data, valid.data); });

    const GpuRect crop = find_crop_v5(valid.data, width, height, heights.data, stacks.data,
                                      rows.data, rect.data);
    if (crop.w <= 0 || crop.h <= 0) return;

    const size_t bytes = static_cast<size_t>(crop.w) * crop.h * 3;
    v1_cuda::Buffer<uint8_t> cropped(bytes);
    const dim3 crop_block(PANORAMA_V4_WARP_BLOCK_X, PANORAMA_V4_WARP_BLOCK_Y);
    const dim3 crop_grid((crop.w + crop_block.x - 1) / crop_block.x,
                         (crop.h + crop_block.y - 1) / crop_block.y);
    v1_cuda::launch([&] { crop_image_kernel<<<crop_grid, crop_block>>>(
        canvas.data, width, height, cropped.data, crop.w, crop.h, crop.x, crop.y); });

    Image result;
    result.w = crop.w; result.h = crop.h; result.c = 3; result.data.resize(bytes);
    v1_cuda::copy(result.data.data(), cropped.data, bytes, cudaMemcpyDeviceToHost);
    pano = std::move(result);
}
} // namespace v5

// Authors: Team 8; implementation assisted by OpenAI Codex.
// V4: two separable Gaussian passes, cooperative shared-memory tiles + halo.
// Coefficients, accumulation order, copies and synchronization follow V1/V3.
#include "../core_headers/v4_stages.h"
#include "../core_headers/cuda_utils.cuh"
#include <climits>
#include <cmath>

// Compile all comparisons with identical flags, except these optional tile
// dimensions when explicitly tuning V4. No runtime/autotuning work is timed.
#ifndef PANORAMA_V4_TILE_X
#define PANORAMA_V4_TILE_X 16
#endif
#ifndef PANORAMA_V4_TILE_Y
#define PANORAMA_V4_TILE_Y 16
#endif

namespace {
constexpr int bx = PANORAMA_V4_TILE_X, by = PANORAMA_V4_TILE_Y;
constexpr int radius = params::kGaussRadius;
static_assert(bx > 0 && by > 0 && bx * by <= 1024, "Invalid V4 block dimensions");

__device__ int clamp_index(int x, int n) { return x < 0 ? 0 : (x >= n ? n - 1 : x); }

// Same GPU coefficient generation as v1_stage12.cu, including operation order
// and global-memory storage. Constant-memory coefficients are a separate idea.
__global__ void gaussian_weights_kernel(float* weights) {
    if (blockIdx.x || threadIdx.x) return;
    float sum = 0.0f;
    for (int i = -radius; i <= radius; ++i) {
        weights[i + radius] = expf(-static_cast<float>(i * i) /
                                  (2.0f * params::kGaussSigma * params::kGaussSigma));
        sum += weights[i + radius];
    }
    for (int i = 0; i < 2 * radius + 1; ++i) weights[i] /= sum;
}

template<bool horizontal>
__global__ void gaussian_blur_tiled_kernel(const float* input, float* output,
                                          int width, int height, const float* weights) {
    constexpr int sw = bx + (horizontal ? 2 * radius : 0);
    constexpr int sh = by + (horizontal ? 0 : 2 * radius);
    __shared__ float tile[sh][sw];
    const int tid = threadIdx.y * bx + threadIdx.x;
    const int ox = blockIdx.x * bx, oy = blockIdx.y * by;
    // Linear cooperative loads cover the core and halo, including partial
    // blocks and images smaller than the radius. All shared entries are set.
    for (int i = tid; i < sw * sh; i += bx * by) {
        const int x = clamp_index(ox + i % sw - (horizontal ? radius : 0), width);
        const int y = clamp_index(oy + i / sw - (horizontal ? 0 : radius), height);
        tile[i / sw][i % sw] = input[static_cast<size_t>(y) * width + x];
    }
    __syncthreads(); // No early returns before this block-wide barrier.
    const int x = ox + threadIdx.x, y = oy + threadIdx.y;
    if (x >= width || y >= height) return;
    float value = 0.0f;
    for (int d = -radius; d <= radius; ++d)
        value += weights[d + radius] *
                 tile[threadIdx.y + (horizontal ? 0 : d + radius)]
                     [threadIdx.x + (horizontal ? d + radius : 0)];
    output[static_cast<size_t>(y) * width + x] = value;
}

void finish_kernel() {
    v1_cuda::check(cudaGetLastError());
    v1_cuda::check(cudaDeviceSynchronize());
}
} // namespace

void v4_gaussian_blur(const Gray& in, Gray& out) {
    out = Gray{};
    if (!valid_gray_image(in)) return;
    const size_t pixels = static_cast<size_t>(in.w) * in.h;
    if (pixels > static_cast<size_t>(INT_MAX))
        throw std::length_error("CUDA V4 blur: image too large");
    v1_cuda::Buffer<float> input(pixels), temporary(pixels), output(pixels);
    v1_cuda::check(v1_cuda::copy(input.data, in.data.data(), pixels * sizeof(float), cudaMemcpyHostToDevice));
    v1_cuda::Buffer<float> weights(2 * radius + 1);
    v1_cuda::launch([&] { gaussian_weights_kernel<<<1, 1>>>(weights.data); });
    finish_kernel();
    const dim3 block(bx, by), grid((in.w + bx - 1) / bx, (in.h + by - 1) / by);
    v1_cuda::launch([&] { gaussian_blur_tiled_kernel<true><<<grid, block>>>(
        input.data, temporary.data, in.w, in.h, weights.data); });
    finish_kernel();
    v1_cuda::launch([&] { gaussian_blur_tiled_kernel<false><<<grid, block>>>(
        temporary.data, output.data, in.w, in.h, weights.data); });
    finish_kernel();
    out.w = in.w; out.h = in.h; out.data.resize(pixels);
    v1_cuda::check(v1_cuda::copy(out.data.data(), output.data, pixels * sizeof(float), cudaMemcpyDeviceToHost));
}

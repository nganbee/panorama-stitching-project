// File: v1_stage12.cu
// Authors: Team 8
// Description: V1 basic CUDA implementation of S1 (grayscale + Gaussian blur)
//              and S2 (Harris response + NMS + top-K).
//
// This version prioritizes a clear correctness baseline. Kernels use regular
// global-memory accesses and one output pixel per thread. Shared-memory tiling,
// kernel fusion and persistent workspaces are reserved for later versions.

#include "../core_headers/v1_stages.h"
#include "../core_headers/cuda_utils.cuh"
#include <cuda_runtime.h>
#include <algorithm>
#include <climits>
#include <cmath>
#include <cfloat>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr int kThreads = 256;

void check(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA S1/S2: ") + operation + ": " +
                                 cudaGetErrorString(status));
    }
}

template <typename T>
struct DeviceBuffer {
    T* data = nullptr;

    explicit DeviceBuffer(std::size_t count = 0) {
        if (count != 0) check(cudaMalloc(reinterpret_cast<void**>(&data), count * sizeof(T)),
                             "cudaMalloc");
    }

    ~DeviceBuffer() {
        if (data != nullptr) cudaFree(data);
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
};

int checked_count(std::size_t count) {
    if (count > static_cast<std::size_t>(INT_MAX))
        throw std::length_error("CUDA S1/S2: image is too large for the baseline kernels");
    return static_cast<int>(count);
}

int one_dimensional_blocks(int count) {
    return (count + kThreads - 1) / kThreads;
}

__device__ int clamp_index(int value, int upper_bound) {
    return value < 0 ? 0 : (value >= upper_bound ? upper_bound - 1 : value);
}

__device__ float sample_clamped(const float* image, int width, int height, int x, int y) {
    const int cx = clamp_index(x, width);
    const int cy = clamp_index(y, height);
    return image[static_cast<std::size_t>(cy) * width + cx];
}

__global__ void rgb_to_gray_kernel(const uint8_t* rgb, float* gray, int count) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const std::size_t base = static_cast<std::size_t>(i) * 3;
    gray[i] = 0.299f * rgb[base] + 0.587f * rgb[base + 1] + 0.114f * rgb[base + 2];
}

template <bool horizontal>
__global__ void gaussian_blur_kernel(const float* input, float* output, int width, int height,
                                     const float* gaussian_weights) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    float value = 0.0f;
    for (int d = -params::kGaussRadius; d <= params::kGaussRadius; ++d) {
        const int sx = horizontal ? x + d : x;
        const int sy = horizontal ? y : y + d;
        value += gaussian_weights[d + params::kGaussRadius] *
                 sample_clamped(input, width, height, sx, sy);
    }
    output[static_cast<std::size_t>(y) * width + x] = value;
}

__global__ void tensor_kernel(const float* blurred,
                              float* ixx, float* iyy, float* ixy,
                              int width, int height) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    const float ix = ((sample_clamped(blurred, width, height, x + 1, y - 1) +
                      2.0f * sample_clamped(blurred, width, height, x + 1, y) +
                      sample_clamped(blurred, width, height, x + 1, y + 1)) -
                     (sample_clamped(blurred, width, height, x - 1, y - 1) +
                      2.0f * sample_clamped(blurred, width, height, x - 1, y) +
                      sample_clamped(blurred, width, height, x - 1, y + 1))) * 0.125f;
    const float iy = ((sample_clamped(blurred, width, height, x - 1, y + 1) +
                      2.0f * sample_clamped(blurred, width, height, x, y + 1) +
                      sample_clamped(blurred, width, height, x + 1, y + 1)) -
                     (sample_clamped(blurred, width, height, x - 1, y - 1) +
                      2.0f * sample_clamped(blurred, width, height, x, y - 1) +
                      sample_clamped(blurred, width, height, x + 1, y - 1))) * 0.125f;

    const std::size_t index = static_cast<std::size_t>(y) * width + x;
    ixx[index] = ix * ix;
    iyy[index] = iy * iy;
    ixy[index] = ix * iy;
}

__global__ void harris_response_kernel(const float* ixx, const float* iyy, const float* ixy,
                                       float* response, int width, int height) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    float sxx = 0.0f, syy = 0.0f, sxy = 0.0f;
    for (int dy = -params::kHarrisWinRadius; dy <= params::kHarrisWinRadius; ++dy) {
        for (int dx = -params::kHarrisWinRadius; dx <= params::kHarrisWinRadius; ++dx) {
            const int sx = clamp_index(x + dx, width);
            const int sy = clamp_index(y + dy, height);
            const std::size_t index = static_cast<std::size_t>(sy) * width + sx;
            sxx += ixx[index];
            syy += iyy[index];
            sxy += ixy[index];
        }
    }

    const float determinant = sxx * syy - sxy * sxy;
    const float trace = sxx + syy;
    response[static_cast<std::size_t>(y) * width + x] =
        determinant - params::kHarrisK * trace * trace;
}

__global__ void max_response_kernel(const float* input, int count, float* partial) {
    __shared__ float values[kThreads];
    float value = 0.0f; // V0 uses rmax = 0, even if every response is negative.
    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < count; i += blockDim.x * gridDim.x) {
        value = fmaxf(value, input[i]);
    }
    values[threadIdx.x] = value;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride)
            values[threadIdx.x] = fmaxf(values[threadIdx.x], values[threadIdx.x + stride]);
        __syncthreads();
    }
    if (threadIdx.x == 0) partial[blockIdx.x] = values[0];
}

__global__ void nms_kernel(const float* response, const float* maximum,
                           Keypoint* candidates, int* candidate_count,
                           int width, int height) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int border = params::kBorder;
    const int radius = params::kNmsRadius;
    if (x < border || y < border || x >= width - border || y >= height - border) return;

    const int index = y * width + x;
    const float value = response[index];
    const float threshold = params::kHarrisThresh * maximum[0];
    if (value <= threshold) return;

    for (int dy = -radius; dy <= radius; ++dy) {
        for (int dx = -radius; dx <= radius; ++dx) {
            if (dx == 0 && dy == 0) continue;
            const int nx = x + dx;
            const int ny = y + dy;
            const int neighbour = ny * width + nx;
            if (response[neighbour] > value ||
                (response[neighbour] == value && neighbour < index)) return;
        }
    }

    const int slot = atomicAdd(candidate_count, 1);
    candidates[slot] = {static_cast<float>(x), static_cast<float>(y), value};
}

__global__ void gaussian_weights_kernel(float* weights) {
    if (blockIdx.x || threadIdx.x) return;
    float sum = 0.0f;
    for (int i = -params::kGaussRadius; i <= params::kGaussRadius; ++i) {
        weights[i + params::kGaussRadius] =
            expf(-static_cast<float>(i * i) /
                     (2.0f * params::kGaussSigma * params::kGaussSigma));
        sum += weights[i + params::kGaussRadius];
    }
    for (int i = 0; i < 2 * params::kGaussRadius + 1; ++i) weights[i] /= sum;
}

__device__ bool better_keypoint(const Keypoint& a, const Keypoint& b) {
    if (a.score != b.score) return a.score > b.score;
    if (a.y != b.y) return a.y < b.y;
    return a.x < b.x;
}

// Global-memory merge sort: each candidate independently finds its rank in
// the other sorted run. One thread per candidate, no host sorting/selection.
__global__ void merge_candidates_kernel(const Keypoint* input, Keypoint* output,
                                         size_t n, size_t width) {
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const size_t base = (i / (2 * width)) * (2 * width);
    const size_t middle = base + width < n ? base + width : n;
    const size_t end = base + 2 * width < n ? base + 2 * width : n;
    const bool left = i < middle;
    size_t lo = left ? middle : base;
    size_t hi = left ? end : middle;
    const size_t other_start = lo;
    while (lo < hi) {
        const size_t mid = lo + (hi - lo) / 2;
        const bool precedes = better_keypoint(input[mid], input[i]) ||
            (!left && !better_keypoint(input[i], input[mid]));
        if (precedes) lo = mid + 1; else hi = mid;
    }
    output[base + (i - (left ? base : middle)) + (lo - other_start)] = input[i];
}

dim3 pixel_block() { return dim3(16, 16); }

dim3 pixel_grid(int width, int height) {
    const dim3 block = pixel_block();
    return dim3((width + block.x - 1) / block.x,
                (height + block.y - 1) / block.y);
}

void finish_kernel(const char* operation) {
    check(cudaGetLastError(), operation);
    check(cudaDeviceSynchronize(), operation);
}

} // namespace

void v1_rgb_to_gray(const Image& in, Gray& gray) {
    gray = Gray{};
    if (!valid_rgb_image(in)) return;

    const std::size_t pixels = static_cast<std::size_t>(in.w) * in.h;
    const int count = checked_count(pixels);
    DeviceBuffer<uint8_t> input(pixels * 3);
    DeviceBuffer<float> output(pixels);

    check(v1_cuda::copy(input.data, in.data.data(), pixels * 3 * sizeof(uint8_t),
                     cudaMemcpyHostToDevice), "grayscale H2D");
    v1_cuda::launch([&] { rgb_to_gray_kernel<<<one_dimensional_blocks(count), kThreads>>>(input.data, output.data, count); });
    finish_kernel("grayscale kernel");

    gray.w = in.w;
    gray.h = in.h;
    gray.data.resize(pixels);
    check(v1_cuda::copy(gray.data.data(), output.data, pixels * sizeof(float),
                     cudaMemcpyDeviceToHost), "grayscale D2H");
}

void v1_gaussian_blur(const Gray& in, Gray& out) {
    out = Gray{};
    if (!valid_gray_image(in)) return;

    const std::size_t pixels = static_cast<std::size_t>(in.w) * in.h;
    checked_count(pixels);
    DeviceBuffer<float> input(pixels);
    DeviceBuffer<float> temporary(pixels);
    DeviceBuffer<float> output(pixels);

    check(v1_cuda::copy(input.data, in.data.data(), pixels * sizeof(float),
                     cudaMemcpyHostToDevice), "blur H2D");
    DeviceBuffer<float> weights(2 * params::kGaussRadius + 1);
    v1_cuda::launch([&] { gaussian_weights_kernel<<<1, 1>>>(weights.data); });
    finish_kernel("Gaussian weights kernel");

    const dim3 block = pixel_block();
    const dim3 grid = pixel_grid(in.w, in.h);
    v1_cuda::launch([&] { gaussian_blur_kernel<true><<<grid, block>>>(input.data, temporary.data, in.w, in.h, weights.data); });
    finish_kernel("horizontal blur kernel");
    v1_cuda::launch([&] { gaussian_blur_kernel<false><<<grid, block>>>(temporary.data, output.data, in.w, in.h, weights.data); });
    finish_kernel("vertical blur kernel");

    out.w = in.w;
    out.h = in.h;
    out.data.resize(pixels);
    check(v1_cuda::copy(out.data.data(), output.data, pixels * sizeof(float),
                     cudaMemcpyDeviceToHost), "blur D2H");
}

void v1_harris_response(const Gray& blurred, Gray& response) {
    response = Gray{};
    if (!valid_gray_image(blurred)) return;

    const std::size_t pixels = static_cast<std::size_t>(blurred.w) * blurred.h;
    checked_count(pixels);
    DeviceBuffer<float> input(pixels);
    DeviceBuffer<float> ixx(pixels), iyy(pixels), ixy(pixels), output(pixels);

    check(v1_cuda::copy(input.data, blurred.data.data(), pixels * sizeof(float),
                     cudaMemcpyHostToDevice), "Harris H2D");
    const dim3 block = pixel_block();
    const dim3 grid = pixel_grid(blurred.w, blurred.h);
    v1_cuda::launch([&] { tensor_kernel<<<grid, block>>>(input.data, ixx.data, iyy.data, ixy.data,
                                   blurred.w, blurred.h); });
    finish_kernel("Sobel/tensor kernel");
    v1_cuda::launch([&] { harris_response_kernel<<<grid, block>>>(ixx.data, iyy.data, ixy.data, output.data,
                                            blurred.w, blurred.h); });
    finish_kernel("Harris response kernel");

    response.w = blurred.w;
    response.h = blurred.h;
    response.data.resize(pixels);
    check(v1_cuda::copy(response.data.data(), output.data, pixels * sizeof(float),
                     cudaMemcpyDeviceToHost), "Harris D2H");
}

void v1_nms_topk(const Gray& response, std::vector<Keypoint>& keypoints) {
    keypoints.clear();
    if (!valid_gray_image(response)) return;

    const std::size_t pixels = static_cast<std::size_t>(response.w) * response.h;
    const int count = checked_count(pixels);
    const int partial_count = one_dimensional_blocks(count);
    DeviceBuffer<float> input(pixels);
    DeviceBuffer<float> partial(static_cast<std::size_t>(partial_count));
    DeviceBuffer<float> maximum(1);
    DeviceBuffer<Keypoint> candidates(pixels);
    DeviceBuffer<int> candidate_count(1);

    check(v1_cuda::copy(input.data, response.data.data(), pixels * sizeof(float),
                     cudaMemcpyHostToDevice), "NMS H2D");
    check(v1_cuda::clear(candidate_count.data, sizeof(int)), "NMS count memset");

    v1_cuda::launch([&] { max_response_kernel<<<partial_count, kThreads>>>(input.data, count, partial.data); });
    finish_kernel("first response reduction");
    v1_cuda::launch([&] { max_response_kernel<<<1, kThreads>>>(partial.data, partial_count, maximum.data); });
    finish_kernel("second response reduction");

    const dim3 block = pixel_block();
    const dim3 grid = pixel_grid(response.w, response.h);
    v1_cuda::launch([&] { nms_kernel<<<grid, block>>>(input.data, maximum.data, candidates.data,
                                candidate_count.data, response.w, response.h); });
    finish_kernel("NMS kernel");

    int count_candidates = 0;
    check(v1_cuda::copy(&count_candidates, candidate_count.data, sizeof(int),
                     cudaMemcpyDeviceToHost), "NMS count D2H");
    if (count_candidates < 0 || count_candidates > count)
        throw std::runtime_error("CUDA S1/S2: invalid NMS candidate count");

    if (!count_candidates) return;
    DeviceBuffer<Keypoint> scratch(static_cast<size_t>(count_candidates));
    Keypoint* source = candidates.data;
    Keypoint* target = scratch.data;
    for (size_t width = 1; width < static_cast<size_t>(count_candidates); width *= 2) {
        v1_cuda::launch([&] { merge_candidates_kernel<<<one_dimensional_blocks(count_candidates), kThreads>>>(
            source, target, count_candidates, width); });
        finish_kernel("GPU keypoint merge sort");
        std::swap(source, target); // swap buffer handles only; no CPU keypoint computation
    }
    const int output_count = count_candidates < params::kTopK ? count_candidates : params::kTopK;
    keypoints.resize(static_cast<size_t>(output_count));
    check(v1_cuda::copy(keypoints.data(), source, keypoints.size() * sizeof(Keypoint),
                     cudaMemcpyDeviceToHost), "sorted top-K D2H");
}

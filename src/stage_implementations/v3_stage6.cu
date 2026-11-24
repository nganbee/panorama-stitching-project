// V3 changes only the histogram row scan used to find the crop rectangle.
// Authors: Team 8; implementation assisted by OpenAI Codex.
#include "../core_headers/v3_stages.h"
#ifndef PANORAMA_STAGE6_HOST_TEST
#include "../core_headers/cuda_utils.cuh"
#endif

#ifdef __CUDACC__
#define STAGE6_HD __host__ __device__
#else
#define STAGE6_HD
#endif

namespace stage6_crop {
// Each entry packs a height and its earliest valid column into one word.
constexpr int kStackFieldBits = 16;
constexpr uint32_t kStackFieldMask = (uint32_t{1} << kStackFieldBits) - 1;
static_assert(stage6_detail::kMaxCanvasDimension <= kStackFieldMask,
              "Canvas dimensions must fit in a crop stack entry");
static_assert(sizeof(int) == sizeof(uint32_t), "Both row scans must share the same scratch size");

STAGE6_HD inline Rect scan_row_compressed(const int* heights, int width, int y, uint32_t* stack) {
    Rect best{};
    int stack_size = 0;
    // Keep the top entry in a local variable. Zero represents an empty stack.
    uint32_t top = 0;
    int top_height = 0;
    for (int x = 0; x <= width; ++x) {
        const int current = x < width ? heights[x] : 0;
        int left = x;
        while (current < top_height) {
            left = static_cast<int>(top & kStackFieldMask);
            const Rect candidate{left, y - top_height + 1, x - left, top_height};
            if (better(candidate, best)) best = candidate;
            top = stack_size > 0 ? stack[--stack_size] : 0;
            top_height = static_cast<int>(top >> kStackFieldBits);
        }
        // Equal heights already share the earliest start. Skip zero-height bars.
        if (current > top_height) {
            if (top_height > 0) stack[stack_size++] = top;
            top = (static_cast<uint32_t>(current) << kStackFieldBits) | static_cast<uint32_t>(left);
            top_height = current;
        }
    }
    return best;
}
}

#undef STAGE6_HD

#ifndef PANORAMA_STAGE6_HOST_TEST
namespace {
__global__ void find_crop_rows_compressed_stack_kernel(
    const int* __restrict__ heights, int width, int height,
    uint32_t* __restrict__ stacks, stage6_crop::Rect* __restrict__ rows)
{
    const int y = blockIdx.x * blockDim.x + threadIdx.x;
    if (y >= height) return;
    const size_t offset = static_cast<size_t>(y) * width;
    rows[y] = stage6_crop::scan_row_compressed(heights + offset, width, y, stacks + offset);
}
}

void stage6_detail::launch_compressed_crop_rows(const int* heights, int width, int height,
                                                void* scratch, stage6_crop::Rect* rows) {
    constexpr int threads = 128;
    v1_cuda::launch([&] { find_crop_rows_compressed_stack_kernel<<<(height + threads - 1) / threads, threads>>>(
        heights, width, height, static_cast<uint32_t*>(scratch), rows); });
}

void v3_warp_blend(const Image& A, const Image& B, const Homography& H, Image& pano) {
    stage6_detail::warp_blend_with_crop_scan(A, B, H, pano, stage6_detail::launch_compressed_crop_rows);
}
#endif

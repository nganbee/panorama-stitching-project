#pragma once

// Shared by the CUDA crop kernels and their portable correctness test.
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

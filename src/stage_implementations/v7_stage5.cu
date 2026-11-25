// File: v7_stage5.cu
// Authors: Team 8 (Thụ: V7 owner)
// Description: V7 S5 RANSAC: one warp scores one homography hypothesis.
// External sources: none (CUDA warp intrinsics from the CUDA C++ Programming Guide).
//
// What changes compared with V1/V5 S5 (v1_stage5.cu):
//   * V1 `score_hypotheses_kernel` uses ONE THREAD per hypothesis and loops over
//     all n matches sequentially: only 2000 threads (~63 warps) are launched,
//     far too few to fill the GPU, and each thread does n projections.
//   * V7 splits that kernel in two:
//       1. `estimate_hypotheses_v7_kernel`: one thread per hypothesis runs the
//          same normalized 4-point DLT as V1 (unchanged math).
//       2. `score_hypotheses_warp_v7_kernel`: ONE WARP per hypothesis. Lane l
//          projects matches l, l+32, l+64, ... (coalesced src/dst loads), the
//          inlier test is reduced with __ballot_sync + __popc, so the warp
//          handles 32 matches per step instead of 1.
//   * `refine_best_warp_v7_kernel` keeps V1's refinement logic but uses the
//     same warp scheme for its inlier work: warp argmax over the 2000
//     candidates, ordered warp compaction of the inliers of the best H, and a
//     warp ballot count of the inliers of the refined H. The least-squares DLT
//     on all inliers still runs in one thread, in V0 order.
//
// What is kept (so inliers, H and the panorama equal V5/V0):
//   * validate/compact/sample kernels and the MT19937 sample sequence (2000
//     iterations, seed, duplicate rejection) are the V1 code.
//   * threshold, error formula, --fmad=false arithmetic.
//   * The summed inlier error used for tie-breaking is accumulated IN MATCH
//     ORDER: after each ballot, every lane walks the set bits of the ballot
//     mask from low to high and adds the shuffled error of that lane. Adding
//     the same doubles in the same order gives the exact V1 sum (no tree
//     reduction, which would change rounding and could flip a tie-break).
//   * Best-candidate rule: more inliers, then lower error, then earlier
//     iteration. A warp reduction with this lexicographic order (index is
//     unique) returns exactly the candidate V1's sequential scan returns.

// Reuse V1's portable S5 math (MT19937, sample_four, normalized DLT, error)
// without its kernels/entry points. This is the same mechanism the V1 host
// test uses; v1_stage5.cu itself is not modified.
#define PANORAMA_STAGE5_HOST_TEST
#include "v1_stage5.cu"
#undef PANORAMA_STAGE5_HOST_TEST

#include "../core_headers/v7_stages.h"
#include "../core_headers/cuda_utils.cuh"
#include <cuda_runtime.h>
#include <climits>
#include <cmath>
#include <cstddef>

namespace {
using stage5::Point;
constexpr double kThresholdSquared = double(params::kRansacThresh) * params::kRansacThresh;
constexpr unsigned kFullMask = 0xffffffffu;
constexpr int kWarpSize = 32;
constexpr int kWarpsPerBlock = 4;             // 128 threads per block
constexpr int kThreadsPerBlock = kWarpSize * kWarpsPerBlock;

// Same layouts as V1 (private there): candidate H, inlier count, summed error.
struct Candidate { Homography H{}; int count = -1; double error = HUGE_VAL; };
struct Result { Homography H{}; int inliers = 0; int ok = 0; };

static_assert(sizeof(V5HomographyResult) == sizeof(Result) &&
              offsetof(V5HomographyResult, H) == offsetof(Result, H) &&
              offsetof(V5HomographyResult, inliers) == offsetof(Result, inliers) &&
              offsetof(V5HomographyResult, ok) == offsetof(Result, ok),
              "V5HomographyResult must match the V7 S5 Result layout");

// ---------------------------------------------------------------------------
// Unchanged V1 kernels (copied verbatim because V1 keeps them file-private).
// ---------------------------------------------------------------------------
__global__ void validate_matches_kernel(const Keypoint* a, int na, const Keypoint* b, int nb,
                                         const Match* matches, int n, int* valid) {
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= size_t(n)) return;
    const Match m = matches[i];
    valid[i] = m.idxA >= 0 && m.idxA < na && m.idxB >= 0 && m.idxB < nb &&
        stage5::finite(a[m.idxA].x) && stage5::finite(a[m.idxA].y) &&
        stage5::finite(b[m.idxB].x) && stage5::finite(b[m.idxB].y);
}

__global__ void compact_points_kernel(const Keypoint* a, const Keypoint* b,
                                       const Match* matches, const int* valid, int n,
                                       Point* src, Point* dst, int* count) {
    if (blockIdx.x || threadIdx.x) return;
    int size = 0;
    for (int i = 0; i < n; ++i) if (valid[i]) {
        const Match m = matches[i];
        src[size] = {b[m.idxB].x, b[m.idxB].y}; // B -> A
        dst[size] = {a[m.idxA].x, a[m.idxA].y};
        ++size;
    }
    *count = size;
}

__global__ void generate_samples_kernel(const int* count, int* samples) {
    if (blockIdx.x || threadIdx.x || *count < 4) return;
    stage5::Mt19937 rng(params::kRansacSeed);
    for (int i = 0; i < params::kRansacIters; ++i)
        stage5::sample_four(rng, *count, samples + 4 * i);
}

// ---------------------------------------------------------------------------
// V7 kernels.
// ---------------------------------------------------------------------------

// Step 1 of V1's score kernel, unchanged: one thread per hypothesis runs the
// 4-point DLT. count = 0 / error = 0 marks a valid hypothesis (V1 sets the same
// values right before its scoring loop); a failed DLT keeps count = -1.
__global__ void estimate_hypotheses_v7_kernel(const Point* src, const Point* dst, const int* count,
                                              const int* samples, Candidate* candidates) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= params::kRansacIters) return;
    Candidate candidate{};
    if (*count >= 4 && stage5::estimate(src, dst, samples + 4 * i, 4, candidate.H.h)) {
        candidate.count = 0; candidate.error = 0;
    }
    candidates[i] = candidate;
}

// Warp-cooperative inlier score of homography h over n matches. All 32 lanes
// must call it (warp-uniform arguments). Returns the inlier count; `error_sum`
// receives the summed squared error of the inliers, accumulated in match order
// (bit-identical to V1's sequential `candidate.error += error`).
__device__ int warp_score(const double* h, const Point* src, const Point* dst, int n,
                          int lane, double& error_sum) {
    int inliers = 0;
    double sum = 0;
    for (int base = 0; base < n; base += kWarpSize) {
        const int j = base + lane;
        double error = HUGE_VAL;
        if (j < n) error = stage5::error(h, src[j], dst[j]);
        unsigned mask = __ballot_sync(kFullMask, j < n && error <= kThresholdSquared);
        inliers += __popc(mask);
        // Ordered accumulation: walk inlier lanes from low to high index.
        while (mask) {
            const int k = __ffs(mask) - 1;
            mask &= mask - 1;
            sum += __shfl_sync(kFullMask, error, k);
        }
    }
    error_sum = sum;
    return inliers;
}

// Step 2: ONE WARP per hypothesis. The candidate H is read by every lane
// (broadcast load), matches are read coalesced (lane l -> match base + l).
__global__ void score_hypotheses_warp_v7_kernel(const Point* src, const Point* dst,
                                                const int* count, Candidate* candidates) {
    const int lane = threadIdx.x & (kWarpSize - 1);
    const int hypothesis = blockIdx.x * kWarpsPerBlock + (threadIdx.x / kWarpSize);
    if (hypothesis >= params::kRansacIters) return;         // warp-uniform
    if (candidates[hypothesis].count < 0) return;           // failed DLT, warp-uniform
    double h[9];
    for (int k = 0; k < 9; ++k) h[k] = candidates[hypothesis].H.h[k];
    double error_sum = 0;
    const int inliers = warp_score(h, src, dst, *count, lane, error_sum);
    if (lane == 0) {
        candidates[hypothesis].count = inliers;
        candidates[hypothesis].error = error_sum;
    }
}

// Lexicographic "better" used by V1's scan: more inliers, then lower error,
// then earlier iteration. Invalid entries (index < 0) never win.
__device__ bool better(int count, double error, int index,
                       int best_count, double best_error, int best_index) {
    if (index < 0) return false;
    if (best_index < 0) return true;
    if (count != best_count) return count > best_count;
    if (error != best_error) return error < best_error;
    return index < best_index;
}

// Refinement, launched as a single warp (<<<1, 32>>>).
__global__ void refine_best_warp_v7_kernel(const Point* src, const Point* dst, const int* count,
                                           const Candidate* candidates, int* selected,
                                           Result* result) {
    const int lane = threadIdx.x;
    if (blockIdx.x || lane >= kWarpSize) return;
    if (lane == 0) *result = Result{};

    // (a) Warp argmax over the 2000 candidates. Each lane scans its strided
    // subset in increasing order, then a butterfly reduction keeps the best.
    int best = -1, best_count = -1;
    double best_error = HUGE_VAL;
    for (int i = lane; i < params::kRansacIters; i += kWarpSize) {
        const Candidate& c = candidates[i];
        if (c.count < 4) continue;
        if (better(c.count, c.error, i, best_count, best_error, best)) {
            best = i; best_count = c.count; best_error = c.error;
        }
    }
    for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
        const int other = __shfl_xor_sync(kFullMask, best, offset);
        const int other_count = __shfl_xor_sync(kFullMask, best_count, offset);
        const double other_error = __shfl_xor_sync(kFullMask, best_error, offset);
        if (better(other_count, other_error, other, best_count, best_error, best)) {
            best = other; best_count = other_count; best_error = other_error;
        }
    }
    if (best < 0) return;                                   // warp-uniform

    // (b) Ordered warp compaction of the inliers of the best hypothesis:
    // selected[] gets exactly the indices, in the same order, as V1.
    const int n = *count;
    double h[9];
    for (int k = 0; k < 9; ++k) h[k] = candidates[best].H.h[k];
    int selected_count = 0;
    for (int base = 0; base < n; base += kWarpSize) {
        const int j = base + lane;
        const bool inlier = j < n && stage5::error(h, src[j], dst[j]) <= kThresholdSquared;
        const unsigned mask = __ballot_sync(kFullMask, inlier);
        if (inlier) selected[selected_count + __popc(mask & ((1u << lane) - 1u))] = j;
        selected_count += __popc(mask);
    }
    __syncwarp();                                           // selected[] visible to lane 0

    // (c) Unchanged refinement: least-squares DLT on all inliers, one thread,
    // V0 accumulation order. Written straight into result->H like V1.
    int refined = 0;
    if (lane == 0) refined = stage5::estimate(src, dst, selected, selected_count, result->H.h);
    refined = __shfl_sync(kFullMask, refined, 0);
    __syncwarp();                                           // result->H visible to all lanes
    if (!refined) return;

    // (d) Warp ballot count of the inliers of the refined H.
    for (int k = 0; k < 9; ++k) h[k] = result->H.h[k];
    double unused = 0;
    const int inliers = warp_score(h, src, dst, n, lane, unused);
    if (lane == 0) {
        result->inliers = inliers;
        result->ok = inliers >= 4;
    }
}
} // namespace

namespace v7 {

bool ransac_homography(const Keypoint* kpA, int na, const Keypoint* kpB, int nb,
                       const Match* matches, int n,
                       V5HomographyResult* device_result, V5HomographyResult& host_result) {
    host_result = V5HomographyResult{};
    if (n < 4 || na <= 0 || nb <= 0) return false;
    v1_cuda::Buffer<int> valid(n), count(1), samples(4 * params::kRansacIters), selected(n);
    v1_cuda::Buffer<Point> src(n), dst(n);
    v1_cuda::Buffer<Candidate> candidates(params::kRansacIters);
    Result* result = reinterpret_cast<Result*>(device_result);
    const int blocks = static_cast<int>((size_t(n) + 127) / 128);
    // Unchanged V1/V5 preparation: validation, ordered compaction, samples.
    v1_cuda::launch([&] { validate_matches_kernel<<<blocks, 128>>>(
        kpA, na, kpB, nb, matches, n, valid.data); });
    v1_cuda::launch([&] { compact_points_kernel<<<1, 1>>>(
        kpA, kpB, matches, valid.data, n, src.data, dst.data, count.data); });
    v1_cuda::launch([&] { generate_samples_kernel<<<1, 1>>>(count.data, samples.data); });
    // V7: thread-per-hypothesis DLT, then warp-per-hypothesis scoring.
    v1_cuda::launch([&] { estimate_hypotheses_v7_kernel<<<(params::kRansacIters + 63) / 64, 64>>>(
        src.data, dst.data, count.data, samples.data, candidates.data); });
    constexpr int score_blocks = (params::kRansacIters + kWarpsPerBlock - 1) / kWarpsPerBlock;
    v1_cuda::launch([&] { score_hypotheses_warp_v7_kernel<<<score_blocks, kThreadsPerBlock>>>(
        src.data, dst.data, count.data, candidates.data); });
    v1_cuda::launch([&] { refine_best_warp_v7_kernel<<<1, kWarpSize>>>(
        src.data, dst.data, count.data, candidates.data, selected.data, result); });
    // Same control/report copy as V5; S6 keeps reading H from device_result.
    v1_cuda::copy(&host_result, device_result, sizeof(V5HomographyResult), cudaMemcpyDeviceToHost);
    return host_result.ok != 0;
}

} // namespace v7

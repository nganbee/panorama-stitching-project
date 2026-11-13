// V1 S5: deterministic GPU sampling, normalized DLT, RANSAC and refinement.
#include "../core_headers/v1_stages.h"
#ifndef PANORAMA_STAGE5_HOST_TEST
#include "../core_headers/cuda_utils.cuh"
#endif
#include <cstdint>

#ifdef __CUDACC__
#define S5_HD __host__ __device__
#else
#define S5_HD
#endif

namespace stage5 {
// Preserve V0's std::mt19937 sequence. One GPU thread generates all samples
// in order, including rejected duplicates, before parallel hypothesis scoring.
struct Mt19937 {
    uint32_t state[624];
    int position = 624;
    S5_HD explicit Mt19937(uint32_t seed) {
        state[0] = seed;
        for (uint32_t i = 1; i < 624; ++i)
            state[i] = 1812433253u * (state[i - 1] ^ (state[i - 1] >> 30)) + i;
    }
    S5_HD uint32_t next() {
        if (position == 624) {
            for (int i = 0; i < 624; ++i) {
                const uint32_t value = (state[i] & 0x80000000u) |
                                       (state[(i + 1) % 624] & 0x7fffffffu);
                state[i] = state[(i + 397) % 624] ^ (value >> 1) ^
                           ((value & 1u) ? 0x9908b0dfu : 0u);
            }
            position = 0;
        }
        uint32_t value = state[position++];
        value ^= value >> 11;
        value ^= (value << 7) & 0x9d2c5680u;
        value ^= (value << 15) & 0xefc60000u;
        return value ^ (value >> 18);
    }
};

// std::uniform_int_distribution is library-specific. Match the libstdc++
// toolchain used by V0/Colab; GCC <=10 and GCC >=11 use different downscaling.
// Fail explicitly on other STLs rather than silently change V0's sample order.
#if !defined(__GLIBCXX__)
#error "S5 sample parity requires libstdc++ (build V0 and V1 with nvcc + GCC, as on Colab)"
#endif
S5_HD inline int pick(Mt19937& rng, int count) {
    const uint32_t range = static_cast<uint32_t>(count);
#if defined(_GLIBCXX_RELEASE) && _GLIBCXX_RELEASE >= 11
    uint64_t product = static_cast<uint64_t>(rng.next()) * range;
    uint32_t low = static_cast<uint32_t>(product);
    if (low < range) {
        const uint32_t threshold = (uint32_t{0} - range) % range;
        while (low < threshold) {
            product = static_cast<uint64_t>(rng.next()) * range;
            low = static_cast<uint32_t>(product);
        }
    }
    return static_cast<int>(product >> 32);
#else
    const uint32_t scaling = UINT32_MAX / range;
    const uint32_t past = range * scaling;
    uint32_t value;
    do { value = rng.next(); } while (value >= past);
    return static_cast<int>(value / scaling);
#endif
}

S5_HD inline void sample_four(Mt19937& rng, int count, int* sample) {
    for (int size = 0; size < 4;) {
        const int candidate = pick(rng, count);
        bool duplicate = false;
        for (int j = 0; j < size; ++j) duplicate = duplicate || sample[j] == candidate;
        if (!duplicate) sample[size++] = candidate;
    }
}
} // namespace stage5
#undef S5_HD

#include <cfloat>
#include <cmath>

#ifdef __CUDACC__
#define S5_HD __host__ __device__
#else
#define S5_HD
#endif

// Portable fixed-buffer implementation of V0's normalized 8-parameter DLT.
// Production V1 calls these helpers only inside GPU kernels. Host annotations
// let the numerical contract be checked on machines without a CUDA compiler.
namespace stage5 {
constexpr double kEpsilon = 1e-12;
struct Point { double x, y; };
struct Normalization { double T[9], invT[9]; };
S5_HD inline bool finite(double value) { return value >= -DBL_MAX && value <= DBL_MAX; }

S5_HD inline bool solve(double* a, double* b) {
    for (int col = 0; col < 8; ++col) {
        int pivot = col;
        double pivot_abs = fabs(a[col * 8 + col]);
        for (int row = col + 1; row < 8; ++row) {
            const double value = fabs(a[row * 8 + col]);
            if (value > pivot_abs) { pivot = row; pivot_abs = value; }
        }
        if (!finite(pivot_abs) || pivot_abs < kEpsilon) return false;
        if (pivot != col) {
            for (int j = col; j < 8; ++j) {
                const double tmp = a[col * 8 + j];
                a[col * 8 + j] = a[pivot * 8 + j]; a[pivot * 8 + j] = tmp;
            }
            const double tmp = b[col]; b[col] = b[pivot]; b[pivot] = tmp;
        }
        const double inv_pivot = 1.0 / a[col * 8 + col];
        for (int j = col; j < 8; ++j) a[col * 8 + j] *= inv_pivot;
        b[col] *= inv_pivot;
        for (int row = 0; row < 8; ++row) {
            if (row == col) continue;
            const double factor = a[row * 8 + col];
            if (fabs(factor) < kEpsilon) continue;
            for (int j = col; j < 8; ++j) a[row * 8 + j] -= factor * a[col * 8 + j];
            b[row] -= factor * b[col];
        }
    }
    return true;
}

S5_HD inline Normalization normalization(const Point* points, const int* selected, int count) {
    double cx = 0, cy = 0;
    for (int i = 0; i < count; ++i) { cx += points[selected[i]].x; cy += points[selected[i]].y; }
    cx /= count; cy /= count;
    double mean_distance = 0;
    for (int i = 0; i < count; ++i) {
        const double dx = points[selected[i]].x - cx, dy = points[selected[i]].y - cy;
        mean_distance += sqrt(dx * dx + dy * dy);
    }
    mean_distance /= count;
    const double scale = mean_distance > kEpsilon ? sqrt(2.0) / mean_distance : 1.0;
    return {{scale, 0, -scale * cx, 0, scale, -scale * cy, 0, 0, 1},
            {1.0 / scale, 0, cx, 0, 1.0 / scale, cy, 0, 0, 1}};
}

S5_HD inline void multiply(const double* a, const double* b, double* out) {
    double tmp[9]{};
    for (int row = 0; row < 3; ++row)
        for (int col = 0; col < 3; ++col)
            for (int k = 0; k < 3; ++k) tmp[row * 3 + col] += a[row * 3 + k] * b[k * 3 + col];
    for (int i = 0; i < 9; ++i) out[i] = tmp[i];
}

S5_HD inline bool estimate(const Point* src, const Point* dst, const int* selected,
                           int count, double* h) {
    if (count < 4) return false;
    const Normalization ns = normalization(src, selected, count);
    const Normalization nd = normalization(dst, selected, count);
    double ata[64]{}, atb[8]{};
    for (int i = 0; i < count; ++i) {
        const Point source = src[selected[i]], dest = dst[selected[i]];
        const Point s{ns.T[0] * source.x + ns.T[2], ns.T[4] * source.y + ns.T[5]};
        const Point d{nd.T[0] * dest.x + nd.T[2], nd.T[4] * dest.y + nd.T[5]};
        const double rows[2][8] = {
            {s.x, s.y, 1, 0, 0, 0, -d.x * s.x, -d.x * s.y},
            {0, 0, 0, s.x, s.y, 1, -d.y * s.x, -d.y * s.y}};
        const double rhs[2] = {d.x, d.y};
        for (int r = 0; r < 2; ++r)
            for (int c = 0; c < 8; ++c) {
                atb[c] += rows[r][c] * rhs[r];
                for (int dcol = 0; dcol < 8; ++dcol)
                    ata[c * 8 + dcol] += rows[r][c] * rows[r][dcol];
            }
    }
    if (!solve(ata, atb)) return false;
    double hn[9] = {atb[0], atb[1], atb[2], atb[3], atb[4], atb[5], atb[6], atb[7], 1};
    double tmp[9]{};
    multiply(hn, ns.T, tmp); multiply(nd.invT, tmp, h);
    if (!finite(h[8]) || fabs(h[8]) < kEpsilon) return false;
    // Preserve V0's division order (h[8] is normalized last).
    for (int i = 0; i < 9; ++i) h[i] /= h[8];
    for (int i = 0; i < 9; ++i) if (!finite(h[i])) return false;
    return true;
}

S5_HD inline double error(const double* h, Point src, Point dst) {
    const double z = h[6] * src.x + h[7] * src.y + h[8];
    if (fabs(z) < kEpsilon) return HUGE_VAL;
    const double x = (h[0] * src.x + h[1] * src.y + h[2]) / z;
    const double y = (h[3] * src.x + h[4] * src.y + h[5]) / z;
    if (!finite(x) || !finite(y)) return HUGE_VAL;
    const double dx = x - dst.x, dy = y - dst.y;
    return dx * dx + dy * dy;
}
} // namespace stage5
#undef S5_HD

#include <climits>

#ifndef PANORAMA_STAGE5_HOST_TEST
namespace {
using stage5::Point;
constexpr double kThresholdSquared = double(params::kRansacThresh) * params::kRansacThresh;
struct Candidate { Homography H{}; int count = -1; double error = HUGE_VAL; };
struct Result { Homography H{}; int inliers = 0; int ok = 0; };

__global__ void validate_matches_kernel(const Keypoint* a, int na, const Keypoint* b, int nb,
                                         const Match* matches, int n, int* valid) {
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= size_t(n)) return;
    const Match m = matches[i];
    valid[i] = m.idxA >= 0 && m.idxA < na && m.idxB >= 0 && m.idxB < nb &&
        stage5::finite(a[m.idxA].x) && stage5::finite(a[m.idxA].y) &&
        stage5::finite(b[m.idxB].x) && stage5::finite(b[m.idxB].y);
}

// Stable compaction keeps exactly V0's match order. Basic GPU baseline.
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

// One thread per hypothesis. Match scoring inside a hypothesis is sequential;
// the warp-per-hypothesis optimization belongs to V7, not this baseline.
__global__ void score_hypotheses_kernel(const Point* src, const Point* dst, const int* count,
                                        const int* samples, Candidate* candidates) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= params::kRansacIters) return;
    Candidate candidate{};
    if (*count >= 4 && stage5::estimate(src, dst, samples + 4 * i, 4, candidate.H.h)) {
        candidate.count = 0; candidate.error = 0;
        for (int j = 0; j < *count; ++j) {
            const double error = stage5::error(candidate.H.h, src[j], dst[j]);
            if (error <= kThresholdSquared) { ++candidate.count; candidate.error += error; }
        }
    }
    candidates[i] = candidate;
}

__global__ void refine_best_kernel(const Point* src, const Point* dst, const int* count,
                                    const Candidate* candidates, int* selected, Result* result) {
    if (blockIdx.x || threadIdx.x) return;
    *result = Result{};
    int best = -1;
    for (int i = 0; i < params::kRansacIters; ++i) {
        if (candidates[i].count < 4) continue;
        if (best < 0 || candidates[i].count > candidates[best].count ||
            (candidates[i].count == candidates[best].count &&
             candidates[i].error < candidates[best].error)) best = i;
        // Exact ties retain the earliest iteration, like V0.
    }
    if (best < 0) return;
    int n = 0;
    for (int i = 0; i < *count; ++i)
        if (stage5::error(candidates[best].H.h, src[i], dst[i]) <= kThresholdSquared)
            selected[n++] = i;
    if (!stage5::estimate(src, dst, selected, n, result->H.h)) return;
    for (int i = 0; i < *count; ++i)
        if (stage5::error(result->H.h, src[i], dst[i]) <= kThresholdSquared) ++result->inliers;
    result->ok = result->inliers >= 4;
}

int checked_count(size_t n) {
    if (n > INT_MAX) throw std::length_error("CUDA S5: too many elements");
    return static_cast<int>(n);
}
} // namespace

bool v1_ransac_homography(const std::vector<Keypoint>& kpA, const std::vector<Keypoint>& kpB,
                          const std::vector<Match>& matches, Homography& H, int& nInliers) {
    H = Homography{}; nInliers = 0;
    if (matches.size() < 4 || kpA.empty() || kpB.empty()) return false;
    const int na = checked_count(kpA.size()), nb = checked_count(kpB.size());
    const int n = checked_count(matches.size());
    v1_cuda::Buffer<Keypoint> a(na), b(nb);
    v1_cuda::Buffer<Match> m(n);
    v1_cuda::Buffer<int> valid(n), count(1), samples(4 * params::kRansacIters), selected(n);
    v1_cuda::Buffer<Point> src(n), dst(n);
    v1_cuda::Buffer<Candidate> candidates(params::kRansacIters);
    v1_cuda::Buffer<Result> result(1);
    v1_cuda::copy(a.data, kpA.data(), kpA.size() * sizeof(Keypoint), cudaMemcpyHostToDevice);
    v1_cuda::copy(b.data, kpB.data(), kpB.size() * sizeof(Keypoint), cudaMemcpyHostToDevice);
    v1_cuda::copy(m.data, matches.data(), matches.size() * sizeof(Match), cudaMemcpyHostToDevice);
    const int blocks = static_cast<int>((size_t(n) + 127) / 128);
    v1_cuda::launch([&] { validate_matches_kernel<<<blocks, 128>>>(
        a.data, na, b.data, nb, m.data, n, valid.data); });
    v1_cuda::launch([&] { compact_points_kernel<<<1, 1>>>(
        a.data, b.data, m.data, valid.data, n, src.data, dst.data, count.data); });
    v1_cuda::launch([&] { generate_samples_kernel<<<1, 1>>>(count.data, samples.data); });
    v1_cuda::launch([&] { score_hypotheses_kernel<<<(params::kRansacIters + 63) / 64, 64>>>(
        src.data, dst.data, count.data, samples.data, candidates.data); });
    v1_cuda::launch([&] { refine_best_kernel<<<1, 1>>>(
        src.data, dst.data, count.data, candidates.data, selected.data, result.data); });
    Result host{};
    v1_cuda::copy(&host, result.data, sizeof(Result), cudaMemcpyDeviceToHost);
    H = host.H; nInliers = host.inliers;
    return host.ok != 0;
}

#endif // PANORAMA_STAGE5_HOST_TEST

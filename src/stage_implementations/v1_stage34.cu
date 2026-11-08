// Baseline CUDA S3/S4: global memory, one thread per keypoint/query.
#include "../core_headers/v1_stage34.h"
#include <cuda_runtime.h>
#include <climits>
#include <stdexcept>
#include <string>

namespace {
void check(cudaError_t status) {
    if (status != cudaSuccess)
        throw std::runtime_error(std::string("CUDA S3/S4: ") + cudaGetErrorString(status));
}
template<class T> struct Buffer {
    T* p = nullptr;
    explicit Buffer(size_t n) { if (n) check(cudaMalloc(reinterpret_cast<void**>(&p), n * sizeof(T))); }
    ~Buffer() { if (p) cudaFree(p); }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
};
int count(size_t n) {
    if (n > INT_MAX) throw std::length_error("CUDA S3/S4: too many elements");
    return static_cast<int>(n);
}
int blocks(int n) { return static_cast<int>((static_cast<size_t>(n) + 127) / 128); }
__device__ uint32_t next_random(uint32_t& state) {
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return state;
}
__device__ int clip(int v, int hi) { return v < 0 ? 0 : (v > hi ? hi : v); }

__global__ void describe_kernel(const float* gray, int w, int h,
                                const Keypoint* kps, int n, Descriptor* out,
                                int* invalid) {
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= static_cast<size_t>(n)) return;
    const Keypoint kp = kps[i];
    if (!isfinite(kp.x) || !isfinite(kp.y) || kp.x < 0 || kp.x >= w || kp.y < 0 || kp.y >= h) {
        atomicExch(invalid, 1);
        return;
    }
    Descriptor d{};
    uint32_t state = params::kBriefSeed;
    const int radius = params::kPatchSize / 2;
    const int cx = static_cast<int>(kp.x), cy = static_cast<int>(kp.y);
    // Each thread regenerates the fixed V0 pattern on device. No CPU pattern
    // generation, shared memory, or constant-memory optimization is required.
    for (int bit = 0; bit < 256; ++bit) {
        const int ax = clip(cx + static_cast<int>(next_random(state) % (2 * radius + 1)) - radius, w - 1);
        const int ay = clip(cy + static_cast<int>(next_random(state) % (2 * radius + 1)) - radius, h - 1);
        const int bx = clip(cx + static_cast<int>(next_random(state) % (2 * radius + 1)) - radius, w - 1);
        const int by = clip(cy + static_cast<int>(next_random(state) % (2 * radius + 1)) - radius, h - 1);
        if (gray[static_cast<size_t>(ay) * w + ax] < gray[static_cast<size_t>(by) * w + bx])
            d.bits[bit / 64] |= uint64_t{1} << (bit % 64);
    }
    out[i] = d;
}
__device__ int distance(const Descriptor& a, const Descriptor& b) {
    int d = 0;
    for (int k = 0; k < 4; ++k)
        d += __popcll(static_cast<unsigned long long>(a.bits[k] ^ b.bits[k]));
    return d;
}
__global__ void nearest_kernel(const Descriptor* query, int nq,
                               const Descriptor* candidates, int nc,
                               int* best, int* first, int* second) {
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= static_cast<size_t>(nq)) return;
    int index = -1, d1 = 257, d2 = 257;
    for (int j = 0; j < nc; ++j) {
        const int d = distance(query[i], candidates[j]);
        // Ascending traversal preserves the lowest-index tie rule of V0.
        if (d < d1) { d2 = d1; d1 = d; index = j; }
        else if (d < d2) d2 = d;
    }
    best[i] = index;
    if (first) first[i] = d1;
    if (second) second[i] = d2;
}
__global__ void filter_kernel(int n, const int* forward, const int* reverse,
                              const int* first, const int* second, int* keep) {
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= static_cast<size_t>(n)) return;
    const int j = forward[i];
    keep[i] = j >= 0 && second[i] != 257 &&
        static_cast<float>(first[i]) < params::kRatio * static_cast<float>(second[i]) &&
        reverse[j] == static_cast<int>(i);
}
// Simple baseline compaction on GPU; preserves ascending idxA order exactly.
__global__ void compact_kernel(int n, const int* keep, const int* best,
                               const int* dist, Match* out, int* size) {
    if (blockIdx.x || threadIdx.x) return;
    int k = 0;
    for (int i = 0; i < n; ++i)
        if (keep[i]) out[k++] = Match{i, best[i], dist[i]};
    *size = k;
}
} // namespace

void v1_describe(const Gray& blurred, const std::vector<Keypoint>& kps,
                 std::vector<Descriptor>& desc) {
    desc.clear();
    if (!valid_gray_image(blurred) || kps.empty()) return;
    const int n = count(kps.size());
    Buffer<float> gray(blurred.data.size());
    Buffer<Keypoint> points(n);
    Buffer<Descriptor> result(n);
    Buffer<int> invalid(1);
    check(cudaMemcpy(gray.p, blurred.data.data(), blurred.data.size() * sizeof(float), cudaMemcpyHostToDevice));
    check(cudaMemcpy(points.p, kps.data(), kps.size() * sizeof(Keypoint), cudaMemcpyHostToDevice));
    check(cudaMemset(invalid.p, 0, sizeof(int)));
    describe_kernel<<<blocks(n), 128>>>(gray.p, blurred.w, blurred.h, points.p, n, result.p, invalid.p);
    check(cudaGetLastError());
    int bad = 0;
    check(cudaMemcpy(&bad, invalid.p, sizeof(int), cudaMemcpyDeviceToHost));
    if (bad) return;
    std::vector<Descriptor> host(n);
    check(cudaMemcpy(host.data(), result.p, host.size() * sizeof(Descriptor), cudaMemcpyDeviceToHost));
    desc.swap(host);
}

void v1_match(const std::vector<Descriptor>& a, const std::vector<Descriptor>& b,
              std::vector<Match>& matches) {
    matches.clear();
    if (a.empty() || b.empty()) return;
    const int na = count(a.size()), nb = count(b.size());
    Buffer<Descriptor> da(na), db(nb);
    Buffer<int> forward(na), reverse(nb), first(na), second(na), keep(na), size(1);
    Buffer<Match> result(na);
    check(cudaMemcpy(da.p, a.data(), a.size() * sizeof(Descriptor), cudaMemcpyHostToDevice));
    check(cudaMemcpy(db.p, b.data(), b.size() * sizeof(Descriptor), cudaMemcpyHostToDevice));
    nearest_kernel<<<blocks(na), 128>>>(da.p, na, db.p, nb, forward.p, first.p, second.p);
    check(cudaGetLastError());
    nearest_kernel<<<blocks(nb), 128>>>(db.p, nb, da.p, na, reverse.p, nullptr, nullptr);
    check(cudaGetLastError());
    filter_kernel<<<blocks(na), 128>>>(na, forward.p, reverse.p, first.p, second.p, keep.p);
    check(cudaGetLastError());
    compact_kernel<<<1, 1>>>(na, keep.p, forward.p, first.p, result.p, size.p);
    check(cudaGetLastError());
    int n = 0;
    check(cudaMemcpy(&n, size.p, sizeof(int), cudaMemcpyDeviceToHost));
    std::vector<Match> host(n);
    if (n) check(cudaMemcpy(host.data(), result.p, host.size() * sizeof(Match), cudaMemcpyDeviceToHost));
    matches.swap(host);
}

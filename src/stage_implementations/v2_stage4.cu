// File: v2_stage4.cu
// Authors: Nguyen Duy Khanh; implementation assisted by OpenAI Codex.
// External sources: none; inherits matching semantics from the team's V1.
// V2 S4: shared-memory descriptor tiling; one thread per query.
#include "../core_headers/v2_stages.h"
#include "../core_headers/cuda_utils.cuh"
#include <cuda_runtime.h>
#include <climits>
#include <stdexcept>
#include <string>

namespace {
void check(cudaError_t status) {
    if (status != cudaSuccess)
        throw std::runtime_error(std::string("CUDA V2 S4: ") + cudaGetErrorString(status));
}
template<class T> struct Buffer {
    T* p = nullptr;
    explicit Buffer(size_t n) { if (n) check(cudaMalloc(reinterpret_cast<void**>(&p), n * sizeof(T))); }
    ~Buffer() { if (p) cudaFree(p); }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
};
int count(size_t n) {
    if (n > INT_MAX) throw std::length_error("CUDA V2 S4: too many elements");
    return static_cast<int>(n);
}
int blocks(int n) { return static_cast<int>((static_cast<size_t>(n) + 127) / 128); }
__device__ int distance(const Descriptor& a, const Descriptor& b) {
    int d = 0;
    for (int k = 0; k < 4; ++k)
        d += __popcll(static_cast<unsigned long long>(a.bits[k] ^ b.bits[k]));
    return d;
}
// All 128 threads participate in both barriers, including inactive queries in
// the final block. Candidate traversal stays ascending across tile boundaries.
__global__ void nearest_tiled_kernel(const Descriptor* query, int nq,
                                     const Descriptor* candidates, int nc,
                                     int* best, int* first, int* second) {
    __shared__ Descriptor tile[128]; // 4096 bytes; same descriptor layout as V1.
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const bool active = i < static_cast<size_t>(nq);
    Descriptor q{};
    if (active) q = query[i];
    int index = -1, d1 = 257, d2 = 257;
    for (size_t base = 0; base < static_cast<size_t>(nc); base += 128) {
        const size_t j = base + threadIdx.x;
        if (j < static_cast<size_t>(nc)) tile[threadIdx.x] = candidates[j];
        __syncthreads();
        const size_t remaining = static_cast<size_t>(nc) - base;
        const int length = remaining < 128 ? static_cast<int>(remaining) : 128;
        if (active) {
            for (int k = 0; k < length; ++k) {
                const int d = distance(q, tile[k]);
                if (d < d1) { d2 = d1; d1 = d; index = static_cast<int>(base) + k; }
                else if (d < d2) d2 = d;
            }
        }
        __syncthreads(); // readers finish before the next tile overwrites storage
    }
    if (active) {
        best[i] = index;
        if (first) first[i] = d1;
        if (second) second[i] = d2;
    }
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

void v2_match(const std::vector<Descriptor>& a, const std::vector<Descriptor>& b,
                std::vector<Match>& matches) {
    matches.clear();
    if (a.empty() || b.empty()) return;
    const int na = count(a.size()), nb = count(b.size());
    Buffer<Descriptor> da(na), db(nb);
    Buffer<int> forward(na), reverse(nb), first(na), second(na), keep(na), size(1);
    Buffer<Match> result(na);
    check(v1_cuda::copy(da.p, a.data(), a.size() * sizeof(Descriptor), cudaMemcpyHostToDevice));
    check(v1_cuda::copy(db.p, b.data(), b.size() * sizeof(Descriptor), cudaMemcpyHostToDevice));
    v1_cuda::launch([&] {
        nearest_tiled_kernel<<<blocks(na), 128>>>(da.p, na, db.p, nb, forward.p, first.p, second.p);
    });
    check(cudaGetLastError());
    v1_cuda::launch([&] {
        nearest_tiled_kernel<<<blocks(nb), 128>>>(db.p, nb, da.p, na, reverse.p, nullptr, nullptr);
    });
    check(cudaGetLastError());
    v1_cuda::launch([&] { filter_kernel<<<blocks(na), 128>>>(na, forward.p, reverse.p, first.p, second.p, keep.p); });
    check(cudaGetLastError());
    v1_cuda::launch([&] { compact_kernel<<<1, 1>>>(na, keep.p, forward.p, first.p, result.p, size.p); });
    check(cudaGetLastError());
    int n = 0;
    check(v1_cuda::copy(&n, size.p, sizeof(int), cudaMemcpyDeviceToHost));
    std::vector<Match> host(n);
    if (n) check(v1_cuda::copy(host.data(), result.p, host.size() * sizeof(Match), cudaMemcpyDeviceToHost));
    matches.swap(host);
}

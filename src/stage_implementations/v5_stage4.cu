// File: v5_stage4.cu
// Authors: Team 8 (Thụ: V5 owner)
// Description: V5 device-resident S4 matching, inheriting V2's shared-memory descriptor tiling.
#include "../core_headers/v5_stages.h"
#include "../core_headers/cuda_utils.cuh"
#include <cuda_runtime.h>
#include <climits>
#include <stdexcept>
#include <string>

namespace {
void check(cudaError_t status) {
    if (status != cudaSuccess)
        throw std::runtime_error(std::string("CUDA V5 S4: ") + cudaGetErrorString(status));
}

template <class T>
struct Buffer {
    T* p = nullptr;
    explicit Buffer(size_t n) {
        if (n) check(cudaMalloc(reinterpret_cast<void**>(&p), n * sizeof(T)));
    }
    ~Buffer() {
        if (p) cudaFree(p);
    }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
};

int blocks(int n) { return static_cast<int>((static_cast<size_t>(n) + 127) / 128); }

__device__ int distance(const Descriptor& a, const Descriptor& b) {
    int d = 0;
    for (int k = 0; k < 4; ++k)
        d += __popcll(static_cast<unsigned long long>(a.bits[k] ^ b.bits[k]));
    return d;
}

__global__ void nearest_tiled_kernel(const Descriptor* query, int nq,
                                     const Descriptor* candidates, int nc,
                                     int* best, int* first, int* second) {
    __shared__ Descriptor tile[128];
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
        __syncthreads();
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

__global__ void compact_kernel(int n, const int* keep, const int* best,
                               const int* dist, Match* out, int* size) {
    if (blockIdx.x || threadIdx.x) return;
    int k = 0;
    for (int i = 0; i < n; ++i)
        if (keep[i]) out[k++] = Match{i, best[i], dist[i]};
    *size = k;
}
} // namespace

namespace v5 {
int match(const Descriptor* a, int na, const Descriptor* b, int nb, Match* matches) {
    if (na <= 0 || nb <= 0) return 0;
    Buffer<int> forward(na), reverse(nb), first(na), second(na), keep(na), size(1);
    v1_cuda::launch([&] {
        nearest_tiled_kernel<<<blocks(na), 128>>>(a, na, b, nb, forward.p, first.p, second.p);
    });
    check(cudaGetLastError());
    v1_cuda::launch([&] {
        nearest_tiled_kernel<<<blocks(nb), 128>>>(b, nb, a, na, reverse.p, nullptr, nullptr);
    });
    check(cudaGetLastError());
    v1_cuda::launch([&] { filter_kernel<<<blocks(na), 128>>>(na, forward.p, reverse.p, first.p, second.p, keep.p); });
    check(cudaGetLastError());
    v1_cuda::launch([&] { compact_kernel<<<1, 1>>>(na, keep.p, forward.p, first.p, matches, size.p); });
    check(cudaGetLastError());
    int n = 0; // metadata only
    check(v1_cuda::copy(&n, size.p, sizeof(int), cudaMemcpyDeviceToHost));
    if (n < 0 || n > na) throw std::runtime_error("CUDA V5 S4: invalid match count");
    return n;
}
} // namespace v5

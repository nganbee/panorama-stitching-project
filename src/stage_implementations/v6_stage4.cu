// File: v6_stage4.cu
// Authors: Nguyen Duy Khanh; implementation assisted by OpenAI Codex.
// Description: V6 one warp/query, tiled candidates and deterministic shuffle top-2.
#include "../core_headers/v6_stages.h"
#include "../core_headers/cuda_utils.cuh"
#include <cuda_runtime.h>
#include <climits>
#include <stdexcept>
#include <string>

namespace {
void check(cudaError_t status) {
    if (status != cudaSuccess)
        throw std::runtime_error(std::string("CUDA V6 S4: ") + cudaGetErrorString(status));
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

// Sort by (Hamming distance, candidate index). Equal distances at distinct
// indices occupy both top slots, exactly as the ascending V0/V5 scan does.
using Key = unsigned long long;
__device__ void insert_key(Key value, Key& first, Key& second) {
    if (value < first) { second = first; first = value; }
    else if (value != first && value < second) second = value;
}

// Four queries/block. All threads, including inactive final warps, reach
// both block barriers. Every shuffle has all 32 lanes participating.
__global__ void nearest_warp_kernel(const Descriptor* query, int nq,
                                    const Descriptor* candidates, int nc,
                                    int* best, int* first, int* second) {
    __shared__ Descriptor tile[128];
    const int lane = threadIdx.x & 31;
    const size_t i = static_cast<size_t>(blockIdx.x) * 4 + threadIdx.x / 32;
    const bool active = i < static_cast<size_t>(nq);
    Descriptor q{};
    if (active) q = query[i];
    Key d1 = ~Key{0}, d2 = ~Key{0};
    for (size_t base = 0; base < static_cast<size_t>(nc); base += 128) {
        const size_t j = base + threadIdx.x;
        if (j < static_cast<size_t>(nc)) tile[threadIdx.x] = candidates[j];
        __syncthreads();
        const size_t remaining = static_cast<size_t>(nc) - base;
        const int length = remaining < 128 ? static_cast<int>(remaining) : 128;
        if (active) {
            for (int k = lane; k < length; k += 32) {
                const Key key = (static_cast<Key>(distance(q,tile[k])) << 32) |
                                static_cast<unsigned int>(base+k);
                insert_key(key,d1,d2);
            }
        }
        __syncthreads();
    }
    for (int offset = 16; offset > 0; offset /= 2) {
        const Key other1 = __shfl_down_sync(0xffffffffu,d1,offset);
        const Key other2 = __shfl_down_sync(0xffffffffu,d2,offset);
        if (lane+offset < 32) {
            insert_key(other1,d1,d2);
            insert_key(other2,d1,d2);
        }
    }
    if (active && lane == 0) {
        best[i] = d1 == ~Key{0} ? -1 : static_cast<int>(static_cast<unsigned int>(d1));
        if (first) first[i] = d1 == ~Key{0} ? 257 : static_cast<int>(d1 >> 32);
        if (second) second[i] = d2 == ~Key{0} ? 257 : static_cast<int>(d2 >> 32);
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

namespace v6 {
int match(const Descriptor* a, int na, const Descriptor* b, int nb, Match* matches) {
    if (na <= 0 || nb <= 0) return 0;
    Buffer<int> forward(na), reverse(nb), first(na), second(na), keep(na), size(1);
    v1_cuda::launch([&] {
        nearest_warp_kernel<<<static_cast<unsigned>((static_cast<size_t>(na)+3)/4), 128>>>(a, na, b, nb, forward.p, first.p, second.p);
    });
    check(cudaGetLastError());
    v1_cuda::launch([&] {
        nearest_warp_kernel<<<static_cast<unsigned>((static_cast<size_t>(nb)+3)/4), 128>>>(b, nb, a, na, reverse.p, nullptr, nullptr);
    });
    check(cudaGetLastError());
    v1_cuda::launch([&] { filter_kernel<<<blocks(na), 128>>>(na, forward.p, reverse.p, first.p, second.p, keep.p); });
    check(cudaGetLastError());
    v1_cuda::launch([&] { compact_kernel<<<1, 1>>>(na, keep.p, forward.p, first.p, matches, size.p); });
    check(cudaGetLastError());
    int n = 0; // metadata only
    check(v1_cuda::copy(&n, size.p, sizeof(int), cudaMemcpyDeviceToHost));
    if (n < 0 || n > na) throw std::runtime_error("CUDA V6 S4: invalid match count");
    return n;
}
} // namespace v6

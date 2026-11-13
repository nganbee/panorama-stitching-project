#pragma once
#include "common.h"
#include <cuda_runtime.h>
#include <limits>
#include <stdexcept>
#include <string>

// Synchronous baseline helpers. All operations use the default stream.
namespace v1_cuda {
inline void check(cudaError_t status) {
    if (status != cudaSuccess)
        throw std::runtime_error(std::string("V1 CUDA: ") + cudaGetErrorString(status));
}

template<class T> struct Buffer {
    T* data = nullptr;
    explicit Buffer(size_t n) {
        if (n > std::numeric_limits<size_t>::max() / sizeof(T))
            throw std::length_error("V1 CUDA buffer size overflow");
        if (n) check(cudaMalloc(reinterpret_cast<void**>(&data), n * sizeof(T)));
    }
    ~Buffer() { if (data) cudaFree(data); }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
};

// One profiler per calling CPU thread; no change to the public stage APIs.
struct Profile {
    StageTimes& times;
    double* stage = nullptr;
    cudaEvent_t start = nullptr, stop = nullptr;
    explicit Profile(StageTimes& target) : times(target) {
        check(cudaEventCreate(&start));
        const cudaError_t status = cudaEventCreate(&stop);
        if (status != cudaSuccess) { cudaEventDestroy(start); check(status); }
    }
    ~Profile() { cudaEventDestroy(start); cudaEventDestroy(stop); }
    Profile(const Profile&) = delete;
    Profile& operator=(const Profile&) = delete;
};
inline thread_local Profile* active_profile = nullptr;
struct ProfileScope {
    Profile* previous;
    explicit ProfileScope(Profile& profile) : previous(active_profile) { active_profile = &profile; }
    ~ProfileScope() { active_profile = previous; }
    ProfileScope(const ProfileScope&) = delete;
    ProfileScope& operator=(const ProfileScope&) = delete;
};

template<class Function> inline void timed(Function&& operation, double* category, bool kernel = false) {
    Profile* profile = active_profile;
    if (profile) check(cudaEventRecord(profile->start));
    operation();
    if (profile) {
        check(cudaEventRecord(profile->stop));
        check(cudaEventSynchronize(profile->stop));
        float milliseconds = 0;
        check(cudaEventElapsedTime(&milliseconds, profile->start, profile->stop));
        if (category) *category += milliseconds;
        // Stage fields contain device computation only; copies are separate.
        if (kernel && profile->stage) *profile->stage += milliseconds;
    }
}

template<class Function> inline void launch(Function&& operation) {
    timed([&] { operation(); check(cudaGetLastError()); },
          active_profile ? &active_profile->times.kernel_total : nullptr, true);
    // Standalone stage calls are blocking too; surface asynchronous failures.
    if (!active_profile) check(cudaDeviceSynchronize());
}

inline cudaError_t copy(void* dst, const void* src, size_t bytes, cudaMemcpyKind kind) {
    double* category = nullptr;
    if (active_profile) {
        if (kind == cudaMemcpyHostToDevice) category = &active_profile->times.h2d;
        if (kind == cudaMemcpyDeviceToHost) category = &active_profile->times.d2h;
    }
    if (bytes) timed([&] { check(cudaMemcpy(dst, src, bytes, kind)); }, category);
    return cudaSuccess;
}

inline cudaError_t clear(void* dst, size_t bytes) {
    // Setup is included in end-to-end. It is neither an explicit kernel launch
    // nor a host/device transfer, so do not mislabel it as either category.
    check(cudaMemset(dst, 0, bytes));
    return cudaSuccess;
}
} // namespace v1_cuda

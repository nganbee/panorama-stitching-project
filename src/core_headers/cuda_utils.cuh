// Authors: Team 8, with Codex implementation assistance.
// CUDA ownership and timing helpers. External sources: none.
#pragma once
#include <cuda_runtime.h>
#include <cstddef>
#include <stdexcept>
#include <string>

inline void cuda_check(cudaError_t status, const char* operation) {
    if (status != cudaSuccess)
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
}
#define CUDA_CHECK(operation) cuda_check((operation), #operation)

// Buffers grow on demand and never shrink during a cached V3 run. They own
// exactly one allocation; copying is deliberately disabled.
template<class T> class DeviceBuffer {
public:
    DeviceBuffer() = default;
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    ~DeviceBuffer() { if (ptr_) cudaFree(ptr_); }
    void reserve(std::size_t elements) {
        if (elements <= capacity_) return;
        T* next = nullptr;
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&next), elements * sizeof(T)));
        if (ptr_) cudaFree(ptr_);
        ptr_ = next;
        capacity_ = elements;
    }
    T* get() { return ptr_; }
    const T* get() const { return ptr_; }
private:
    T* ptr_ = nullptr;
    std::size_t capacity_ = 0;
};

// Only device intervals are measured here. Host work is timed separately by
// steady_clock, and memcpy events are accumulated in h2d/d2h, never kernels.
class GpuTimer {
public:
    GpuTimer() {
        CUDA_CHECK(cudaEventCreate(&start_));
        const cudaError_t status = cudaEventCreate(&stop_);
        if (status != cudaSuccess) { cudaEventDestroy(start_); cuda_check(status, "cudaEventCreate"); }
    }
    GpuTimer(const GpuTimer&) = delete;
    GpuTimer& operator=(const GpuTimer&) = delete;
    ~GpuTimer() { cudaEventDestroy(start_); cudaEventDestroy(stop_); }
    void start(cudaStream_t stream = nullptr) { CUDA_CHECK(cudaEventRecord(start_, stream)); }
    double stop(cudaStream_t stream = nullptr) {
        CUDA_CHECK(cudaEventRecord(stop_, stream));
        CUDA_CHECK(cudaEventSynchronize(stop_));
        float elapsed = 0;
        CUDA_CHECK(cudaEventElapsedTime(&elapsed, start_, stop_));
        return elapsed;
    }
private:
    cudaEvent_t start_{}, stop_{};
};

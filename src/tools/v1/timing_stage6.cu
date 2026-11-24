// Compare complete V0/V1 Stage 6 calls, including allocation and transfers.
// Inputs and byte-for-byte validation are outside the timed regions.
#include "../../core_headers/v0_stages.h"
#include "../../core_headers/v1_stages.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using Clock = std::chrono::steady_clock;

void cuda_check(cudaError_t error) {
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}

Image make_image(int w, int h, unsigned seed) {
    Image image;
    image.w = w;
    image.h = h;
    image.data.resize(static_cast<size_t>(w) * h * 3);
    for (size_t i = 0; i < image.data.size(); ++i) {
        seed = seed * 1664525u + 1013904223u;
        image.data[i] = static_cast<uint8_t>(seed >> 24);
    }
    return image;
}

double mean(const std::vector<double>& samples) {
    double sum = 0;
    for (double sample : samples) sum += sample;
    return sum / samples.size();
}

double median(std::vector<double> samples) {
    std::sort(samples.begin(), samples.end());
    const size_t mid = samples.size() / 2;
    return samples.size() % 2 ? samples[mid] : (samples[mid - 1] + samples[mid]) / 2;
}

double measure(bool gpu, const Image& a, const Image& b, const Homography& h) {
    Image output;
    if (gpu) cuda_check(cudaDeviceSynchronize());
    const auto start = Clock::now();
    if (gpu) {
        v1_warp_blend(a, b, h, output);
        cuda_check(cudaGetLastError());
        cuda_check(cudaDeviceSynchronize());
    } else {
        v0_warp_blend(a, b, h, output);
    }
    const double ms = std::chrono::duration<double, std::milli>(Clock::now() - start).count();
    if (!valid_rgb_image(output)) throw std::runtime_error("Stage 6 returned an empty/invalid image");
    return ms;
}

void benchmark(int w, int h, int runs, bool perspective) {
    const Image a = make_image(w, h, 123u), b = make_image(w, h, 456u);
    // The same B -> A transform and input bytes are used by both versions.
    // Scale the perspective coefficients so its strength is similar at every size.
    const Homography transform = perspective
        ? Homography{{1.0, 0.02, w * 0.25 + 0.5, 0.01, 1.0, -h * 0.03 - 0.25,
                      0.02 / w, 0.01 / h, 1.0}}
        : Homography{{1.0, 0.0, w * 0.25 + 0.5, 0.0, 1.0, 0.25, 0.0, 0.0, 1.0}};
    const char* name = perspective ? "perspective" : "translation";
    int out_w = 0, out_h = 0;
    {
        // Untimed warm-up for each implementation, also checking correctness.
        Image cpu, gpu;
        v0_warp_blend(a, b, transform, cpu);
        v1_warp_blend(a, b, transform, gpu);
        cuda_check(cudaGetLastError());
        cuda_check(cudaDeviceSynchronize());
        if (!valid_rgb_image(cpu) || !valid_rgb_image(gpu) ||
            cpu.w != gpu.w || cpu.h != gpu.h || cpu.c != gpu.c || cpu.data != gpu.data) {
            throw std::runtime_error(std::string("V0/V1 output mismatch: ") + name +
                                     " at " + std::to_string(w) + "x" + std::to_string(h));
        }
        out_w = cpu.w;
        out_h = cpu.h;
    }

    std::vector<double> cpu_ms, gpu_ms;
    for (int i = 0; i < runs; ++i) {
        // Alternate order to reduce systematic first/second-run bias.
        if (i % 2 == 0) {
            cpu_ms.push_back(measure(false, a, b, transform));
            gpu_ms.push_back(measure(true, a, b, transform));
        } else {
            gpu_ms.push_back(measure(true, a, b, transform));
            cpu_ms.push_back(measure(false, a, b, transform));
        }
    }
    const double cpu_median = median(cpu_ms), gpu_median = median(gpu_ms);
    std::printf("%d,%d,%s,%d,%d,%d,%.6f,%.6f,%.6f,%.6f,%.6f\n",
                w, h, name, out_w, out_h, runs, mean(cpu_ms), mean(gpu_ms),
                cpu_median, gpu_median, cpu_median / gpu_median);
    std::fflush(stdout);
    std::fprintf(stderr, "Done: %dx%d %s; median V0=%.3f ms, V1=%.3f ms\n",
                 w, h, name, cpu_median, gpu_median);
}
} // namespace

int main(int argc, char** argv) {
    const int runs = argc == 2 ? std::atoi(argv[1]) : 7;
    if (argc > 2 || runs < 5) {
        std::fprintf(stderr, "Usage: %s [runs >= 5]\n", argv[0]);
        return 2;
    }
    try {
        int device = 0;
        cuda_check(cudaGetDevice(&device));
        cudaDeviceProp properties{};
        cuda_check(cudaGetDeviceProperties(&properties, device));
        std::fprintf(stderr, "GPU: %s; timing complete Stage 6 calls (ms)\n", properties.name);
        // Initialize CUDA before the per-case warm-ups and measurements.
        cuda_check(cudaFree(nullptr));
        std::puts("width,height,transform,out_width,out_height,runs,v0_mean_ms,v1_mean_ms,"
                  "v0_median_ms,v1_median_ms,speedup");
        const int sizes[][2] = {{320, 180}, {1280, 720}, {1920, 1080}, {3840, 2160}};
        for (const auto& size : sizes) {
            benchmark(size[0], size[1], runs, false);
            benchmark(size[0], size[1], runs, true);
        }
    } catch (const std::exception& error) {
        std::fprintf(stderr, "Benchmark failed: %s\n", error.what());
        return 1;
    }
    return 0;
}

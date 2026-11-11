// File: run_stage12_cuda_selftest.cpp
// Authors: Team 8
// Description: V0/V1 parity checks for the basic CUDA S1-S2 implementation.

#include "../src/core_headers/v0_stages.h"
#include "../src/core_headers/v1_stages.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <random>
#include <stdexcept>
#include <vector>

namespace {

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

Image make_random_image(int width, int height) {
    Image image;
    image.w = width;
    image.h = height;
    image.c = 3;
    image.data.resize(static_cast<std::size_t>(width) * height * 3);
    std::mt19937 rng(20261111u);
    for (uint8_t& value : image.data) value = static_cast<uint8_t>(rng() & 255u);
    return image;
}

void compare_float_images(const Gray& cpu, const Gray& gpu,
                          float relative_tolerance, float absolute_tolerance,
                          const char* name) {
    require(cpu.w == gpu.w && cpu.h == gpu.h, name);
    require(cpu.data.size() == gpu.data.size(), name);
    float max_error = 0.0f;
    float max_allowed = absolute_tolerance;
    bool within_tolerance = true;
    for (std::size_t i = 0; i < cpu.data.size(); ++i) {
        const float error = std::fabs(cpu.data[i] - gpu.data[i]);
        const float allowed = absolute_tolerance + relative_tolerance * std::fabs(cpu.data[i]);
        max_error = std::max(max_error, error);
        max_allowed = std::max(max_allowed, allowed);
        within_tolerance = within_tolerance && error <= allowed;
    }
    if (!within_tolerance) {
        std::fprintf(stderr, "%s max_error=%.8f allowed=%.8f\n", name, max_error, max_allowed);
        throw std::runtime_error(name);
    }
}

void compare_keypoints(const std::vector<Keypoint>& cpu,
                       const std::vector<Keypoint>& gpu,
                       float score_tolerance) {
    require(cpu.size() == gpu.size(), "S2 keypoint count mismatch");
    for (std::size_t i = 0; i < cpu.size(); ++i) {
        require(cpu[i].x == gpu[i].x && cpu[i].y == gpu[i].y,
                "S2 keypoint coordinate/order mismatch");
        if (std::fabs(cpu[i].score - gpu[i].score) > score_tolerance)
            throw std::runtime_error("S2 keypoint score mismatch");
    }
}

void run_random_parity() {
    const Image image = make_random_image(127, 93);

    Gray cpu_gray, gpu_gray;
    v0_rgb_to_gray(image, cpu_gray);
    v1_rgb_to_gray(image, gpu_gray);
    compare_float_images(cpu_gray, gpu_gray, 1e-6f, 1e-4f, "S1 grayscale mismatch");

    Gray cpu_blur, gpu_blur;
    v0_gaussian_blur(cpu_gray, cpu_blur);
    v1_gaussian_blur(gpu_gray, gpu_blur);
    compare_float_images(cpu_blur, gpu_blur, 2e-6f, 2e-4f, "S1 blur mismatch");

    Gray cpu_response, gpu_response;
    v0_harris_response(cpu_blur, cpu_response);
    // Use the CPU blur for both calls so this check isolates Harris arithmetic
    // from the small floating-point differences produced by the blur kernel.
    v1_harris_response(cpu_blur, gpu_response);
    compare_float_images(cpu_response, gpu_response, 2e-5f, 5e-2f, "S2 Harris mismatch");

    std::vector<Keypoint> cpu_keypoints, gpu_keypoints;
    v0_nms_topk(cpu_response, cpu_keypoints);
    // Feed the same response to both implementations to test NMS and the
    // deterministic host top-K ordering independently of Harris arithmetic.
    v1_nms_topk(cpu_response, gpu_keypoints);
    compare_keypoints(cpu_keypoints, gpu_keypoints, 1e-4f);
}

void run_constant_parity() {
    Image image;
    image.w = 96;
    image.h = 64;
    image.c = 3;
    image.data.assign(static_cast<std::size_t>(image.w) * image.h * 3, 128);

    Gray cpu_gray, gpu_gray, cpu_blur, gpu_blur, cpu_response, gpu_response;
    std::vector<Keypoint> cpu_keypoints, gpu_keypoints;
    v0_rgb_to_gray(image, cpu_gray);
    v1_rgb_to_gray(image, gpu_gray);
    v0_gaussian_blur(cpu_gray, cpu_blur);
    v1_gaussian_blur(gpu_gray, gpu_blur);
    v0_harris_response(cpu_blur, cpu_response);
    v1_harris_response(gpu_blur, gpu_response);
    v0_nms_topk(cpu_response, cpu_keypoints);
    v1_nms_topk(gpu_response, gpu_keypoints);

    compare_float_images(cpu_gray, gpu_gray, 1e-6f, 1e-4f, "constant grayscale mismatch");
    compare_float_images(cpu_blur, gpu_blur, 2e-6f, 2e-4f, "constant blur mismatch");
    compare_float_images(cpu_response, gpu_response, 2e-5f, 5e-2f, "constant Harris mismatch");
    require(cpu_keypoints.empty() && gpu_keypoints.empty(),
            "constant image should have no keypoints");
}

} // namespace

int main() {
    try {
        run_random_parity();
        run_constant_parity();
        std::puts("S1/S2 CUDA parity tests passed");
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "S1/S2 CUDA parity test failed: %s\n", error.what());
        return 1;
    }
}
